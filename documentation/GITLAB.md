# GitLab and more
[List of Documentation Files](menu.md)

## Continuous Integration

Every commit will be tested in gitlab.

See the hidden File: `.gitlab-ci.yml` in the main Folder.

The Command `make test` ist used to run all test scripts from folder test.
You can run this before you push your commit.

Different evaluations scripts are run and the results are saved.
This only works on high-end PCs.

## Testing

there are multiple unit tests to run

### Comparing results

In the new code "evaluation" is splitted in "evaluation" and "train".

Here are some points to make an easy comparison.

#### Probabilites

Direct comparison of propabilites is possible in file *base_evaluation.py*.


| | old (eval-branch) | new (master_duplicate) |
| --- | --- | --- |
| function | weighted_evaluation | weighted_predict_proba (run evaluation) |
| variable | y_hat | ys_predicted |

#### Feature Values

Feature values can be saved with *DataFrame.to_csv('FileName.csv', float_format='%+0.7e', index=False)*

This hast to be done in train function of some method like *evaluation_score_level.py*

| | old (eval-branch) | new (master_duplicate) |
| --- | --- | --- |
| function | train | train (run train) |
| variable | Xs | feature_values |

#### Classifiers

in train (see before) with self.clfs there is access to the classifiers.
Compare fixation and saccade classifiers on theire feature_importance.
If there is somehting different, it will be later.