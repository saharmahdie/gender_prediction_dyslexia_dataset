# Overview of SMIDA
[List of Documentation Files](menu.md)

The scripts to run are in the folder [**scripts**](#scripts).
All the code behind is stored in [**schau_mir_in_die_augen**](#schau_mir_in_die_augen).
The folder [**docker**](#docker) is fore the dockerfile to create uniform virtual enviroments.

The folders **config** and **notebook** are not in use.

<!--- There are three hidden folders (.git, .idea, .vscode). --->

## Content of Folders

### Top Level

| Name | Purpose |
| --- | --- |
| feature_table.txt | list with features we use or known from papers |
| [Makefile](#makefile) | Shortcuts to run the Program |
| Pipfile | ? |
| Pipfile.lock | ? |
| requirements.txt | See [Docker](#docker)|
| .dockerignore | ? |
| .gitattributes | Files which are used with git large file storage |
| .gitignore | Files Excluded from versioning |
| .gitlab-ci.yml | [Continiuos Integration](GITLAB.md) Tasks to automatically test the Code |


### scripts

| Name | Purpose |
| --- | --- |
| dataset_comparison.py | |
| docker_ci.patch | |
| eval_cross.sh | |
| evaluation.py | main [EVALUATION](EVALUATION.md) routine|
| eval_whl.sh | |
| push_image_ci.sh | |
| server_share.sh | |
| startJupyter.sh | |
| warm_cache.py | |

### schau_mir_in_die_augen

In the Top Level there are the files *feature_extraction* and *features* wich define all our features. <!-- todo: maybe move featur-list.txt there -->
Furthermore there is *trajectory_split* with contains *ivt*.

Overview of folders with programcode:

| Folder | Purpose |
| --- | --- |
| datasets | loading of datasets |
| evaluation | Different evaluation [METHODS](METHODS.md) |
| rbfn | Implementation of **R**adial **B**asic **F**unction **N**etworks|
| test | Test routines (for CI) |
| visualizaion | Functions implementing bokeh to show data |

### docker

see [README](/README.md)

todo: more or delete

## Special Files

### Makefile

[Thie Makefile](https://gitlab.informatik.uni-bremen.de/ascadian/schau_mir_in_die_augen/blob/master/Makefile) helps to run scripts with more complicated calls by simple commands.

```
make docs	# creat API Documentation
make bokeh	# start visualization
make test	# run tests
...
```

Run `make help` to see an incomplete list of available commands or look into the file to see all options and get examples for calls.

### Docker

-  Requirements file
