# phony calls will be done, no matter what.
.PHONY: clean clean-model clean-pyc docs help init init-docker create-container start-container jupyter test lint profile clean clean-data clean-docker clean-container clean-image sync-from-source sync-to-source
.DEFAULT_GOAL := help

###########################################################################################################
## SCRIPTS
###########################################################################################################

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
        match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
        if match:
                target, help = match.groups()
                print("%-20s %s" % (target, help))
endef

define START_DOCKER_CONTAINER
if [ `$(DOCKER) inspect -f {{.State.Running}} $(CONTAINER_NAME)` = "false" ] ; then
        $(DOCKER) start $(CONTAINER_NAME)
fi
endef


###########################################################################################################
## VARIABLES
###########################################################################################################

export DOCKER=scripts/dockerwrap.sh
export PRINT_HELP_PYSCRIPT
export START_DOCKER_CONTAINER
export PYTHONPATH=$(shell pwd)
export PROJECT_NAME=schau_mir_in_die_augen
export IMAGE_NAME=$(PROJECT_NAME)-image
export CONTAINER_NAME=$(PROJECT_NAME)-container
export JUPYTER_HOST_PORT=8888
export JUPYTER_CONTAINER_PORT=8888
export PYTHON=python3
export DOCKERFILE=docker/Dockerfile

###########################################################################################################
## ADD TARGETS FOR YOUR TASK
###########################################################################################################


## GENERAL TARGETS
###########################################################################################################

help:
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


bokeh: ## trajectory visualization with bokeh
	@echo "open http://localhost:5006/gui_bokeh in a web browser"
	@echo ""
	cd schau_mir_in_die_augen && bokeh serve visualization/gui_bokeh.py --dev

test: ## run test cases in tests directory
	cd schau_mir_in_die_augen && $(PYTHON) -m unittest discover -t ..

jupyter: ## start Jupyter Notebook server
	jupyter-notebook --ip=0.0.0.0 --port=${JUPYTER_CONTAINER_PORT}

lint: ## check style with flake8
	flake8 schau_mir_in_die_augen

profile: ## show profile of the project
	@echo "CONTAINER_NAME: $(CONTAINER_NAME)"
	@echo "IMAGE_NAME: $(IMAGE_NAME)"
	@echo "JUPYTER_PORT: `$(DOCKER) port $(CONTAINER_NAME)`"

doc docs: ## create documentation
	cd docs && $(MAKE) html
	@echo "open 'http://localhost:63342/SMIDA/docs/build/html/index.html' in a web browser"
	# not sure this link will work for all

## Download Dataset TARGETS
###########################################################################################################
# todo: add other datasets
# todo: git lfs pull necessary? Maybe make this specific for the datasets

dataset-bioeye:
	git submodule update --init --remote -- data/BioEye2015_DevSets
dataset-rigas:
	git submodule update --init --remote -- data/RigasEM
dataset-dyslexia:
	git submodule update --init --remote -- data/Dyslexia
dataset-whl:
	git submodule update --init --remote -- data/where_humans_look
dataset-biometricsDS:
	git submodule update --init --remote -- data/data_cleaned_biometrics_ds

## Docker TARGETS
###########################################################################################################

init: $(SELECT_DOCKER) init-docker create-container ## initialize repository for traning

init-docker: ## initialize docker image
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKERFILE) --build-arg UID=$(shell id -u) .

create-container: ## create docker container
	$(DOCKER) run --ipc=host -it \
	-v $(PWD):/work \
	-p $(JUPYTER_HOST_PORT):$(JUPYTER_CONTAINER_PORT) \
	-p 5006:5006 \
	--name $(CONTAINER_NAME) $(IMAGE_NAME)

start-container: ## start bash in docker container
	@echo "$$START_DOCKER_CONTAINER" | $(SHELL)
	@echo "Launched $(CONTAINER_NAME)..."
	$(DOCKER) exec -it $(CONTAINER_NAME) bash

attach-container: ## attach to first session of container
	@echo "$$START_DOCKER_CONTAINER" | $(SHELL)
	@echo "New session in $(CONTAINER_NAME)..."
	$(DOCKER) attach $(CONTAINER_NAME)

## Cleaning TARGETS
###########################################################################################################

clean: clean-cache clean-model clean-pyc clean-docker ## remove all artifacts
	rm -rf cache/*

clean-cache:
	rm -rf cache/ schau_mir_in_die_augen/.cache/ .cache/ scripts/.cache /tmp/smida-cache

clean-model: ## remove model artifacts
	rm -fr model/*

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

distclean: clean clean-data ## remove all the reproducible resources including Docker images

clean-docker: clean-container clean-image ## remove Docker image and container

clean-container: ## remove Docker container
	-$(DOCKER) stop $(CONTAINER_NAME)
	-$(DOCKER) rm $(CONTAINER_NAME)

clean-image: ## remove Docker image
	-$(DOCKER) image rm $(IMAGE_NAME)

ci-push-image:
	scripts/push_image_ci.sh

############################################################################################################
## Gender prediction paper main experiments accuracies:
Mixed_Group_accuracy:
	cd scripts && /bin/bash paper_gender_prediction_mixed_group_accuracy.sh
Non_dyslexic_Group_accuracy:
	cd scripts && /bin/bash paper_gender_prediction_Non_dyslexic_group_accuracy.sh
Dyslexic_Group_accuracy:
	cd scripts && /bin/bash paper_gender_prediction_Dyslexic_group_accuracy.sh

Mixed_Group_average_accuracy:
	cd scripts && $(PYTHON) paper_gender_prediction_Mixed_group_avarage_accuracy.py
Non_dyslexic_Group_average_accuracy:
	cd scripts && $(PYTHON) paper_gender_prediction_Non_dyslexic_group_avarage_accuracy.py
Dyslexic_Group_average_accuracy:
	cd scripts && $(PYTHON) paper_gender_prediction_Dyslexic_group_avarage_accuracy.py

## Evaluation
###########################################################################################################

# training configurations
eval-our-append:
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran --classifier rbfn

eval-score-level1y:
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --test_dataset bio-tex1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --test_dataset bio-tex1y --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --test_dataset bio-ran1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --test_dataset bio-ran1y --classifier rbfn

eval-our-one-clf:
	cd scripts && $(PYTHON) train.py --method our-one-clf --dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-one-clf --dataset bio-tex --classifier rf

	cd scripts && $(PYTHON) evaluation.py --method our-one-clf --dataset bio-tex --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method our-one-clf --dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-one-clf --dataset bio-ran --classifier rbfn

eval-windowed:
	cd scripts && $(PYTHON) evaluation.py --method our-windowed --dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-windowed --dataset bio-tex --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method our-windowed --dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-windowed --dataset bio-ran --classifier rbfn

eval-one-another-train-1y:
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex1y --test_dataset bio-ran --classifier rf --score_level_1y_train
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex1y --test_dataset bio-ran --classifier rbfn --score_level_1y_train
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran1y --test_dataset bio-tex --classifier rf --score_level_1y_train
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran1y --test_dataset bio-tex --classifier rbfn --score_level_1y_train


# jobs for paper
eval-score-level:
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --classifier rbfn

eval-one-another:
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --test_dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --test_dataset bio-ran --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --test_dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --test_dataset bio-tex --classifier rbfn
	# cross with 1y
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --test_dataset bio-ran1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --test_dataset bio-ran1y --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --test_dataset bio-tex1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --test_dataset bio-tex1y --classifier rbfn

	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran1y --test_dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran1y --test_dataset bio-tex --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex1y --test_dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex1y --test_dataset bio-ran --classifier rbfn

	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --test_dataset bio-tex1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --test_dataset bio-tex1y --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex1y --test_dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex1y --test_dataset bio-tex --classifier rbfn

	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --test_dataset bio-ran1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran --test_dataset bio-ran1y --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran1y --test_dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-ran1y --test_dataset bio-ran --classifier rbfn

eval-one-another-append:
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex --test_dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex --test_dataset bio-ran --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran --test_dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran --test_dataset bio-tex --classifier rbfn
	# cross with 1y
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex --test_dataset bio-ran1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex --test_dataset bio-ran1y --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran --test_dataset bio-tex1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran --test_dataset bio-tex1y --classifier rbfn

	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran1y --test_dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran1y --test_dataset bio-tex --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex1y --test_dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex1y --test_dataset bio-ran --classifier rbfn

	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex --test_dataset bio-tex1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex --test_dataset bio-tex1y --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex1y --test_dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-tex1y --test_dataset bio-tex --classifier rbfn

	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran --test_dataset bio-ran1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran --test_dataset bio-ran1y --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran1y --test_dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method our-append --dataset bio-ran1y --test_dataset bio-ran --classifier rbfn

eval-score-level-whl-1-rf-1:
	cd scripts && /bin/bash eval_whl.sh rf 1 300 score-level

eval-score-level-whl-1-rf-2:
	cd scripts && /bin/bash eval_whl.sh rf 2 300 score-level

eval-score-level-whl-1-rf-3:
	cd scripts && /bin/bash eval_whl.sh rf 3 300 score-level

eval-score-level-whl-1-rbfn:
	cd scripts && /bin/bash eval_whl.sh rbfn 1 300 score-level 1 --use_valid_data
	cd scripts && /bin/bash eval_whl.sh rbfn 2 300 score-level 2 --use_valid_data
	cd scripts && /bin/bash eval_whl.sh rbfn 3 300 score-level 3 --use_valid_data

eval-score-level-whl-2-rf-1:
	cd scripts && /bin/bash eval_whl.sh rf 10 300 score-level

eval-score-level-whl-2-rf-2:
	cd scripts && /bin/bash eval_whl.sh rf 20 300 score-level

eval-score-level-whl-2-rf-3:
	cd scripts && /bin/bash eval_whl.sh rf 30 300 score-level

eval-score-level-whl-2-rbfn:
	cd scripts && /bin/bash eval_whl.sh rbfn 10 300 score-level 10
	cd scripts && /bin/bash eval_whl.sh rbfn 20 300 score-level
	cd scripts && /bin/bash eval_whl.sh rbfn 30 300 score-level

eval-score-level-whl-3-1:
	cd scripts && /bin/bash eval_whl.sh rf 100 300 score-level
	cd scripts && /bin/bash eval_whl.sh rf 200 300 score-level

eval-score-level-whl-3-2:
	cd scripts && /bin/bash eval_whl.sh rbfn 100 300 score-level
	cd scripts && /bin/bash eval_whl.sh rbfn 200 300 score-level

eval-score-level-whl-3-3:
	cd scripts && /bin/bash eval_whl.sh rbfn 300 300 score-level
	cd scripts && /bin/bash eval_whl.sh rf 300 300 score-level

eval-score-level-whl-3-4:
	cd scripts && /bin/bash eval_whl.sh rbfn 500 300 score-level
	cd scripts && /bin/bash eval_whl.sh rf 500 300 score-level

eval-score-level-whl-3-5:
	cd scripts && /bin/bash eval_whl.sh rbfn 700 300 score-level
	cd scripts && /bin/bash eval_whl.sh rf 700 300 score-level

# all paper features
eval-paper-append:
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-tex --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-ran --classifier rbfn

eval-one-another-all-paper:
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-tex --test_dataset bio-ran --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-tex --test_dataset bio-ran --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-ran --test_dataset bio-tex --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-ran --test_dataset bio-tex --classifier rbfn
	# cross with 1y
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-tex --test_dataset bio-ran1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-tex --test_dataset bio-ran1y --classifier rbfn
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-ran --test_dataset bio-tex1y --classifier rf
	cd scripts && $(PYTHON) evaluation.py --method paper-append --dataset bio-ran --test_dataset bio-tex1y --classifier rbfn

eval-paper-append-whl-1-rf-1:
	cd scripts && /bin/bash eval_whl.sh rf 1 300 paper-append

eval-paper-append-whl-1-rf-2:
	cd scripts && /bin/bash eval_whl.sh rf 2 300 paper-append

eval-paper-append-whl-1-rf-3:
	cd scripts && /bin/bash eval_whl.sh rf 3 300 paper-append

eval-paper-append-whl-1-rbfn:
	cd scripts && /bin/bash eval_whl.sh rbfn 1 300 paper-append 1
	cd scripts && /bin/bash eval_whl.sh rbfn 2 300 paper-append 2
	cd scripts && /bin/bash eval_whl.sh rbfn 3 300 paper-append 3

eval-paper-append-whl-2-rf-1:
	cd scripts && /bin/bash eval_whl.sh rf 10 300 paper-append

eval-paper-append-whl-2-rf-2:
	cd scripts && /bin/bash eval_whl.sh rf 20 300 paper-append

eval-paper-append-whl-2-rf-3:
	cd scripts && /bin/bash eval_whl.sh rf 30 300 paper-append

eval-paper-append-whl-2-rbfn:
	cd scripts && /bin/bash eval_whl.sh rbfn 10 300 paper-append 10
	cd scripts && /bin/bash eval_whl.sh rbfn 20 300 paper-append
	cd scripts && /bin/bash eval_whl.sh rbfn 30 300 paper-append

eval-paper-append-whl-3-1:
	cd scripts && /bin/bash eval_whl.sh rf 100 300 paper-append
	cd scripts && /bin/bash eval_whl.sh rf 200 300 paper-append

eval-paper-append-whl-3-2:
	cd scripts && /bin/bash eval_whl.sh rf 100 300 paper-append
	cd scripts && /bin/bash eval_whl.sh rf 200 300 paper-append

eval-paper-append-whl-3-3:
	cd scripts && /bin/bash eval_whl.sh rbfn 300 300 paper-append
	cd scripts && /bin/bash eval_whl.sh rf 300 300 paper-append

eval-paper-append-whl-3-4:
	cd scripts && /bin/bash eval_whl.sh rbfn 500 300 paper-append
	cd scripts && /bin/bash eval_whl.sh rf 500 300 paper-append

eval-paper-append-whl-3-5:
	cd scripts && /bin/bash eval_whl.sh rbfn 700 300 paper-append
	cd scripts && /bin/bash eval_whl.sh rf 700 300 paper-append

## missing evals
train-50-1:
	cd scripts && /bin/bash eval_whl.sh rbfn 50 300 paper-append
train-50-2:
	cd scripts && /bin/bash eval_whl.sh rf 50 300 paper-append
train-50-3:
	cd scripts && /bin/bash eval_whl.sh rbfn 50 300 score-level
train-50-4:
	cd scripts && /bin/bash eval_whl.sh rf 50 300 score-level

train-100-1:
	cd scripts && /bin/bash eval_whl.sh rbfn 100 300 paper-append
train-100-2:
	cd scripts && /bin/bash eval_whl.sh rbfn 200 300 paper-append

train-300-1:
	cd scripts && /bin/bash eval_whl.sh rbfn 300 300 score-level
train-300-2:
	cd scripts && /bin/bash eval_whl.sh rf 300 300 score-level


# cross validation
cross-1-1:
	cd scripts && /bin/bash eval_cross.sh paper-append bio-tex rbfn
cross-1-2:
	cd scripts && /bin/bash eval_cross.sh paper-append bio-tex rf
cross-1-3:
	cd scripts && /bin/bash eval_cross.sh paper-append bio-ran rbfn
cross-1-4:
	cd scripts && /bin/bash eval_cross.sh paper-append bio-ran rf
# score level
cross-2-1:
	cd scripts && /bin/bash eval_cross.sh score-level bio-tex rbfn
cross-2-2:
	cd scripts && /bin/bash eval_cross.sh score-level bio-tex rf
cross-2-3:
	cd scripts && /bin/bash eval_cross.sh score-level bio-ran rbfn
cross-2-4:
	cd scripts && /bin/bash eval_cross.sh score-level bio-ran rf
