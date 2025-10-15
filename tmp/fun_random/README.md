# Fun with Python's random module

```bash
python random.py
```

No need to modify PYTHONPATH or anything.

## Content of random.py

```python
import numpy as np

i = np.random.randint(0, 100)
```

It seems basic stuff, and if you know numpy, you think it is basic code.

## Failure

It create an exception deep into python with a stack trace almost 1000 call long
looping on some code in numpy.

```python
import numpy.random as random
```

## Where is the problem?

After that you start with
> The problem is in the filename, silly, don't call your file random.py and you will be fine.

Yes the bug is in the filename, but the issue is deeper. I am ok to rename this file
but which name should i use ? Which one is safe ?

We known that using a name that is the same as a module is bad. We know the current list of stdlib modules.
OK those i can avoid... Obviously new python versions will add new modules, so i need to avoid those too.
It is ok, they are not modified that often.

But it is not all, i need to avoid **ANY** module names, not just the ones from stdlib. This means
all pip modules too... And those i can not really avoid them, the list is moving all the time.

## Where is the problem ? (my guess)

I think the problem is that `import` doesn't separate the namespace of the stdlib modules, vs the user ones.
In C, it is done by `#include <...>` vs `#include "..."`.
In node.js, they namespaced all the imports from the stdline as `import * as fs from 'node:fs'`.

In python, there is no such separation. So if you have a file named `random.py` in your current directory,
it will shadow the stdlib module `random` but also any other module named `random` that you may have installed
with pip.

Q. is that a bug or a feature ?
Q. is that a security issue ? Could a malicious module can shadow a stdlib one ?

## What is the solution ?

Well as a user, a python beginner as well, i dunno how to work it around 😀
