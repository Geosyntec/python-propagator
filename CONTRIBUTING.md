# Contributing to ``python-propagator``

## General `git` workflow

### Starting out
  1. Fork the main repository to your github account
  2. Clone your fork via `git clone git@github.com:<your_github_name>/python-propagator.git`

You should have at least two git remotes on your local machine:
  1. `origin` (e.g., `git@github.com:phobson/python-propagator.git`)
  1. `Geosyntec` or `upstream` (e.g., `git@github.com:Geosyntec/python-propagator.git`)

You add the upstream fork with:
```
git remote add upstream git@github.com:Geosyntec/python-propagator.git
```

Since you have both of these remotes, we can safely say that you should:
  * never, *never* push to `origin/master`.
  * never, *never*, **never** push to `upstream/master`.
  * only ever push to a feature branch on `origin`.

All changes should be made a feature branch checked out from `upstream/master` (or similar).

### Adding a new feature, fixed bugs, etc

If your local repo is current, the general workflow is this:
```
git checkout -b new-feature # use a better name
## edit analysis.py
git add propagator/this_widget.js
git commit -m "added a new widget that solves problems"
git push origin new-feature
```

OK.
Now go to www.github.com/Geosyntec/python-propagator and create a pull request (PR).
When you create the pull request, add tags if necessary, assign a milestone, and assign someone to *review* your work.
Review is critical here.
Reviewers should at least document any discussions about the PR in the comment thread.
After the reviewer and author are happy with state of the PR it can be merged.
For this project only I can merge things.
I rule this repository with an iron fist.

### Staying current with the `Geosyntec` repo.
While you're away working on some feature in its own branch, others are doing the same.
In fact, there's a good chance that the code base changed while you were working.
After your PR has been merged, you should checkout our local master branch, fetch the `Geosyntec` repo, and do a fast-forward only merge.
```
git checkout master
git fetch Geosyntec
git merge --ff-only Geosyntec/master
```

This works because your origin/master branch has a clean history because you did all of your work in feature branches.

### Summary
  * Never push to any fork other than your own
  * Always work in a feature branch off of the most current `Geosyntec`
  * Open PRs from `origin/feature` to `Geosyntec/master`
  * wait for review before merging.


## Code style guide

### Naming conventions
Let's stick with the general python convention of classes being in `UpperCamelCase_WithSomeUnderscores`.
Also, let's keep functions and variables all lower case with underscores (e.g., `output_layer`).

Names should be brief, but descriptive.
  * Good: `subcatchment_filename`
  * Bad: `subcatchment_inputlayer_provided_by_user`
  * Worse: `filename`

### Documentation
All functions and classes should documented with docstrings and inline comments.

**Everything must be documented**.
If you don't write documentation, I won't merge your PR.

#### Docstrings
All functions *must* be documented with docstrings adherent to the numpydoc standard.
Feel free to read the [full guidance on documentation](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).
But you'd probably be better off just looking at existing examples in the code base and [a full example](https://github.com/numpy/numpy/blob/master/doc/example.py)

#### Inline comments
Inline comments are generally a good idea.
However, I want them to explain *why* you're doing something -- not what you're doing.
*What* you're doing should be obvious.

Bad:
```python
# read a csv file, unstack, compute the median:
medians = pandas.read_csv("input.csv").unstack(level='pollutant').median()
```

Better:
```
# Load the water quality data and pivot the pollutants into columns.
# Then compute the medians of all the columns to characterize the BMP
# performance for each pollutant.
medians = pandas.read_csv("input.csv").unstack(level='pollutant').median()
```

### Line length
Try to keep the lines to ~90 characters long.
Longer lines are fine if they are more readable than breaking a long statement over several lines, but try to keep it reasonable.


### Unit tests
In general, I'd like use to take a test-driven-development approach.
That means that when we decide that we need a function to do something, we *first* write unit tests that specify the behavior of that function with various inputs.
If you don't know what a unit test is, [you can read about them](https://en.wikipedia.org/wiki/Unit_testing).

**Everyting must be tested**.
If you don't write tests, I won't merge your PR.

#### Writing tests
This projet uses the `nose` library to write and run unit tests.

The `utils` module already has a comprehensive set of tests.
Use them as examples.
Your tests should include a minimal input dataset and expected result datasets if it operates on e.g., spatial data.
Tests should confirm that a function:
  1. handles valid and invalid input as expected
  1. raises errors when expected
  1. generates expected output for a range of inputs that we expected to encounter
  1. behaves properly in less likely edge and corner cases (e.g., what happens if the shapefile has no records?)

Tests are run via the command line with the following command:
```
F:\phobson\SOC_WQIP\python-propagator>nosetests
...............................................................................................
Ran 96 tests in 9.950s

OK

If any tests fail, you'll see "F"s instead of dots

```

Use can use the coverage options tell `nose` to print out a little summary of which portions of the code were not run by the tests:
```
...............................................................................................

F:\phobson\SOC_WQIP\python-propagator>nosetests --with-coverage --cover-package=propagator
Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
propagator.py                3      0   100%
propagator\analysis.py      19      6    68%   45, 49, 53, 57, 61, 65
propagator\utils.py        220      3    99%   127-128, 578
------------------------------------------------------
TOTAL                      242      9    96%
----------------------------------------------------------------------
Ran 96 tests in 9.950s

OK
```


## Overall approached
If we're going to test everything, that means that in general, our function should be short and simple.
They should except primitive values (e.g., strings, integers) when possible, and call as few other functions (that we've written) as possible.
Functions like this make up the so-called functional core of the code base (`utils.py`).

Eventually we'll string these functions together in the so-called imperative shell (`analysis.py`).
That file will contain a few high-level functions that rely on the core to execute the workflows needed to perform the analysis.

Testing the imperative shell is a little trickier.
I'll worruy about that.
