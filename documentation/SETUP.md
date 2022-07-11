# SETUP: Schau mir in die Augen
[List of Documentation Files](menu.md)

## Preparation

You need:

-  **Python**
  It is suggested to have **Python3.7** installed.
  See versions listed in the `scripts/inspection.py` file
-  **git** with [**lfs**](https://help.github.com/en/articles/installing-git-large-file-storage) 

| Task | Ubuntu | macOS |
| --------- | --------- | --------- |
| install Git lfs | `sudo apt-get install git-lfs`<br />Some errors during git-lfs installation can be ignored if the installation completed successfully |  ? |

- Windows users: Check [Windows](Windows.md) for help. (e.g. run make commands)

### Downloading Repository and Data

-  clone repository
	 `git clone git@gitlab.informatik.uni-bremen.de:cgvr/smida2/schau_mir_in_die_augen.git`
	 - you can add `--recursive` to initialize all submodules directly
	 - It is necessary that you had generated a SSH-Key and uploaded it to GitLab (see Settings)
-  activate large-file-supoort `git lfs install`
-  get data `git submodule update --init <SUBMOUDLE>`
		add your specific subbmodule names, or leave empty to initialize all.
-  get data for real: go to submodule and run `git lfs pull`

The Data can be downloaded via make:

```makefile
make dataset-bioeye
make dataset-rigas
make dataset-dyslexia
make dataset-whl
```



### Fulfilling Requirements

On **Ubuntu** you can install packages by using `sudo pip install <name of package>`

- `sudo pip install requirements.txt` should install all needed Packages

**Hint**: Make sure to check pythonpath is addressed correctly
`https://stackoverflow.com/questions/11960602/how-to-add-something-to-pythonpath`

Maybe it is helpful to run: `export PYTHONPATH=$PWD` when you are in the SMIDA Repository or set it manual.

### Creating Code Documentation

Run in Terminal

```
cd docs
make html
```

Open `/docs/build/html/index.html` in you Browser.

*Todo: This has to be somwhere else.*

You can change the Documentation by editing `docs/source/XXX.rst`

## Using

### With PyCharm

#### Visualization

You want to Run: **schau_mir_in_die_augen/visualization/gui_bokeh.py** 

In `Edit Configuration` add the following parameters:

```
Module name: bokeh
Parameters: serve gui_bokeh.py --dev
```
Make sure a compatible Python version (e.g. **Python 3.7**) is added and selected in `Edit Configuration`

### With Docker

**Docker** makes sure the code run always in the same enviroment without everytime creating a comlete virtual machine. See [README](/README.md) for more information.

| Task | Ubuntu | macOS |
| --------- | --------- | --------- |
| install docker | `sudo apt-get install docker.io` ([Source](https://askubuntu.com/questions/938700/how-do-i-install-docker-on-ubuntu-16-04-lts)) | [See Here](https://docs.docker.com/docker-for-mac/)|
| sudo docker | Your Account has to be added to the docker group by `sudo usermod -a -G docker $USER` a restart is required ([Source](https://techoverflow.net/2017/03/01/solving-docker-permission-denied-while-trying-to-conn	ect-to-the-docker-daemon-socket/)). |

initialize docker by`make init`
This will automatically log you in to the created container

-  If not already done, log in to docker by
	`make start-container`.
-  Try `make test`. If this works its a first sign everything is fine.
-  Now you can try to run some [EVALUATION](EVALUATION.md).
	See the makefile for ideas or look in [OVERVIEW](OVERVIEW.md)
-  To inspect the data, run `make bokeh`.
-  Log out by
	`exit`.

### Known Errors

-  Ubuntu: SIGKILL
	Happens when memory is not enough and process gets killed.
-  Python Version is an common problem. Maybe you want to run:  
    	`python scripts/inspection.py`
        	and try different Versions.
