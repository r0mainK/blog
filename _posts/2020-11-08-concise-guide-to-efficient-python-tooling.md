---
layout: post
title: "Concise guide to efficient Python tooling"
date: 2020-11-09
---
Although I intend to mostly write about AI, for my first post I am simply going to share my current setup for coding in Python. Tooling and "good practices" is a pretty contentious topic so I don't expect you, dear reader, to agree with everything I'll be covering - hell, I might even disagree with this in a couple years' time. However, I strongly believe that the following makes coding Python much easier so if you haven't thought out much of your tooling then this post might be of interest to you - and even if you have, I'll welcome any criticism or recommendations 👌

Anyway, let's get into it !

## Editor

I've been using [VSCode](https://code.visualstudio.com/) for a while now, and honestly if you haven't I recommend you try it out. The UX is really great, as are the numerous extensions and features you can use. It might not be Python-specific, but I did not find any feature to be lacking when compared to other IDEs like Pycharm or Spyder. In the rest of the post, I'll assume you are using VSCode - that way I can show you how to integrate the other tools directly, using the `.vscode/settings.json` file in your project. Once you've got it installed, go ahead and choose a theme, then install Microsoft's Python extension right away. Incidentally, if you don't want Microsoft to gather your data, then you should add the following to your global settings file:

```json
{
    "telemetry.enableTelemetry": false,
    "telemetry.enableCrashReporter": false
}
```

I would also recommend setting the following option, as I've had some issues with the default language server Jedi:

```json
{
    "python.languageServer": "Microsoft"
}
```

## Dependency management

At this point, I've been working with [Poetry](https://python-poetry.org/) somewhat reluctantly at work. While it does get the job done (better than pipenv or conda at least), I feel it's overkill in most cases, and with the increased usage of containers, it can be cumbersome to use. Anyway, in my opinion try and keep control over dependency management while you still can using the built-in `venv`, and switch to Poetry only if you must.

To keep everything clean, I like to do this the following way. Assuming you are at the root directory of your project run:

```
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
```

Then, add the following items your config file:

```json
{
    "python.pythonPath": ".env/bin/python3",
    "files.exclude": {
        ".env/**": true,
        "**/__pycache__": true,
        "**/**/*.pyc": true
    },
    "files.watcherExclude": {
        ".env/**": true,
        "**/__pycache__": true,
        "**/**/*.pyc": true
    }
}
```

With this, VSCode will know which Python executable should be used, and it won't show or check out files in that directory. I also added files that may be created by Python when running your code, to avoid clutter. 

Finally, I recommend you split all dependencies in two files: `requirements.txt` and `requirements-dev.txt`. The former should contain packages the project depends upon, while the latter should be used for any packages used for development, like testing frameworks, or the tools I'm about to describe. As mentioned earlier, doing things this way means you'll be responsible for dependency management, so when adding anything you should either pin the version, or at least specify major upper and lower limits, e.g.:

```
torch >=1.6,<1.7
```

The issue with this is well known: nested dependencies are not pinned, which will lead to `pip` nagging sooner or later. If you're worried and want to be a bit safer, then you might as well create an environment where only packages from your `requirements.txt` are installed, check everything works, then do something like `pip freeze > requirements.txt` to generate a proper snapshot of your environment. This isn't perfect as you can't be certain packages won't be retconned, but hopefully this should be enough in most cases.

## Version Control

Not much to say here, if you don't use `git` yet you should stop and go check it out (get it ?), then start using it in all of your projects. There are a ton of resources online to help you get the hang of it, I personally like [this lecture](https://missing.csail.mit.edu/2020/version-control/) of the notorious missing semester course from MIT (in fact you should study the whole course if you've got the time). By the way, you should be adding to your `.gitignore` file the file patterns / directories mentioned earlier, to avoid committing temporary files or any other trash by mistake. Additionally, if you don't like using the command line, notice you can use VSCode to do most of the operations.

## Formatting

You know what they say : [black](https://github.com/psf/black) is the new black. More seriously, at the risk of sounding like a fanboy using this has been a game-changer for me, it just takes a load off your back - and pressing `Ctrl-S` becomes so much more satisfying 🤗 To integrate with VSCode, simply add this to your settings after installing it:

```json
{
    "editor.formatOnSave": true,
    "editor.formatOnSaveTimeout": 10000,    
    "python.formatting.blackPath": ".env/bin/black",
    "python.formatting.blackArgs": [
        "--line-length",
        "99"
    ]
}
```

As you can see I specify a different line length then the default (88), but apart from that no configuration is required, as that is the basic purpose of this tool. Setting the timeout to a higher value can be pretty useful if you have many files in your project.

## Sorting imports

Here this is really about preference, I recommend you make up your mind by checking out [this repo](https://github.com/PyCQA/flake8-import-order) - and stick to it by using [isort](https://github.com/PyCQA/isort) to order imports on save. I like the appnexus style, with the additional constraint of keeping only one import per line for clarity. If you want to try it out, here is how to do it:

```json
{
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.sortImports.path": ".env/bin/isort",
    "python.sortImports.args": [
        "--line-width=100",
        "--force-single-line-imports",
        "--force-sort-within-sections",
        "--lines-after-imports=2",
        "--lines-between-types=0",
        "--dont-order-by-type"
    ]
}
```

## Linting

Again, this somewhat comes down to preferences, but I've been using [flake8](https://flake8.pycqa.org/en/latest/) for a while now and have come to grow fond of it. Especially, the fact you can add extensions to it pretty easily has motivated open-source developers to create [a ton](https://github.com/DmytroLitvinov/awesome-flake8-extensions) of them. I personally use [bugbear](https://github.com/PyCQA/flake8-bugbear), [import-order](https://github.com/PyCQA/flake8-import-order) (that I mentioned earlier), [quotes](https://github.com/zheller/flake8-quotes), [builtins](https://github.com/gforcada/flake8-builtins) and I've just discovered [use-fstring](https://github.com/MichaelKim0407/flake8-use-fstring) which I'll definitely be using from here on out 🤩

To use all this, add the following to your configuration: 

```json
{
    "python.linting.enabled": true,
    "python.linting.lintOnSave": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Path": ".env/bin/flake8",
    "python.linting.flake8Args": [
        "--extend-ignore=E731",
        "--max-line-length=99",
        "--import-order-style=appnexus",
        "--inline-quotes=double"
    ]
}
```

As you can see I'm using the `extend-ignore` option to disable error E731, "do not assign a lambda expression". This option is pretty useful if you have specific errors you wish to ignore for some reason. Whilst you shouldn't overuse this, rules should be broken if need be, especially in case of conflicts with other tools, like black for instance.

## Documentation 

I personally don't like in-code documentation, and think many people use documentation as a crutch to avoid naming variables precisely, e.g. this:

```python
def f(x):
    """Apply the softmax function to an input vector x"""
    x = np.exp(x)
    return x / np.sum(x)

```

Instead of this:

```python
def softmax(x):
    x = np.exp(x)
    return x / np.sum(x)
```

Now this is obviously a simple example, and you will sometimes need to document your code. To this end, Nils Werner has written a great extension for VSCode that analyzes your function signature to generate the doc skeleton when you type `"""` and press `Enter`, so you should definitely install and use it. To select a specific docstring format, simply add the following to your settings (I like [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html)):

```json
{
   "autoDocstring.docstringFormat": "numpy"
}
```

## In conclusion

That about wraps it, thanks for reading if you made it this far ! If you've set up the environment and tried applying it to your project, what are your thoughts ? Hit me up, I'll be glad to chat about it ! Also, share it if you want me (or this) to gain clout 😁

Next up should be a post on embeddings, so if you fancy NLP, stay tuned ...
