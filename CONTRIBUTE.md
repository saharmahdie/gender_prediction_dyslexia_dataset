# How to Contribute

Please document your problems and improve the [Documentation](documentation/OVERVIEW.md) at anytime you find something is missing.

This document is a collection of How to write Code in SMIDA. The requirements are not met anywhere in the project. Please improve bad code if you can.

## Code Style

- Please use Type Hints:

```
# this method uses an integer and a string to create a dictionary
def some_method(some_var: int, some_optional_var: str = 'whatever') -> dict:
	...
```

- Please write Comments and Documentation
- Please try to fix all warnings inside PyCharm (Some files ignoring this on purpose)
- Please create specific methods if you use a code block multiple times.
  Don't copy code with minor modifications.
- Don't leave code out-commented to use it maybe later. Make a note in which version of the Repository it can be found.

## Comments and Documentation

General Discussion: [realpython.com](https://realpython.com/documenting-python-code/) and possible Formats: [datacamp](https://www.datacamp.com/community/tutorials/docstrings-python)

We use the [**reStructured** Style](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

###  Default Docstrings

```python
def/class whatever():
"""This is the summary line

This is the further elaboration of the docstring. Within this section,
you can elaborate further on details as appropriate for the situation.
Notice that the summary and the elaboration is separated by a blank new
line.

:param arg1: description
:param arg2: description
:type arg1: type description
:type arg2: type description
:return: return description
:rtype: the return type description

:Example:

followed by a blank line !

.. seealso:: blabla
.. warnings also:: blabla
.. note:: blabla
.. todo:: blabla
"""
```

Please make sure to use the empty lines. They are not optional!

**Class docstrings** should contain the following information:

- **A brief summary of its purpose and behavior**

The class constructor parameters should be documented within the `__init__` class method docstring. Individual methods should be documented using their individual docstrings. Class method docstrings should contain the following:

- **A brief description of what the method is and what it’s used for**
- Any arguments (both required and optional) that are passed including keyword arguments
- Label any arguments that are considered optional or have a default value
- *Any side effects that occur when executing the method*
- *Any exceptions that are raised*
- *Any restrictions on when the method can be called*

### Attributes/Variables/Properties

You can give a Variable a public information for the API Documentation:

```
#: This will appera in the API Documentation
some_variable = 23
# This is only a comment in the code
some_other_variable = 'Whatever'
```

### API Documentation

You can modify the API documentation in `/docs/source/`.

Here are so [Styling Information](https://pythonhosted.org/an_example_pypi_project/sphinx.html).