# The Ultimate Flask Course

Found links to possible code:

https://github.com/PrettyPrinted/intro_to_flask

## Learn the fundamentals of the Flask framework and its various extensions

[Anthony Herbert](https://www.udemy.com/course/the-ultimate-flask-course/#instructor-1)

https://www.udemy.com/course/the-ultimate-flask-course/

# Flask Basics

## Installation on Windows

### python virtual environment

In Windows, `py -m venv env`.

https://docs.python.org/3/library/venv.html

```
python3 -m venv /path/to/new/virtual/environment
```
This is for Mac OS/Linux.

e.g.

```
mkdir -p venv
python3 -m venv ./venv
```

To "activate" this environment,

Windows:
```
env\Scripts\activate
```

To begin using the virtual environment, it needs to be activated:

```
$ source venv/bin/activate
```
e.g. 
```
[topolo@localhost UdemyUltimateFlask]$ python3 -m venv ./virtualenv/
[topolo@localhost UdemyUltimateFlask]$ source ./virtualenv/bin/activate
(virtualenv) [topolo@localhost UdemyUltimateFlask]$ 
```

## 2. The Two ways of Running Flask Apps

1. `flask run` 
*Requires* file name to be `app.py`. Otherwise, must do
```
export FLASK_APP=yourappname.py
```

2.  `python3 app.py`, depending on newer version, might not work well.
```
if __name__ == '__main__':
	app.run()
```

Remember, `export FLASK_DEBUG=1` for debug mode on.

But this is only for local machine, deployment different.

4. Install Using Pipevn

```
pipenv install flask
```
Install `flask` in virtual environment, automatically creates `virtual env` based on location currently in.
```
pipenv shell # to activate.
```


## 5. Intro to Routes

Important because it's how user goes to different parts of app

