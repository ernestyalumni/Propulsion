# Running the Flask app

[Auto reload python Flask app upon code changes](https://stackoverflow.com/questions/16344756/auto-reloading-python-flask-app-upon-code-changes) by doing

```
export FLASK_ENV=development
flask run
```

```
flask shell
```
Use this code to enter shell on the application context.

cf. Gaspar and Stouffer (2018), Ch. 1, Getting Started, pp. 23

Useful information - get all defined URL routes:
```
flask shell

app.url_map
```

Show where Flask thinks our templates and static folders are,
```
app.static_folder
app.has_static_folder
app.template_folder

```

For more ...

# Using Flask's command-line interface

cf. pp. 22-23, Gaspar and Stouffer (2018), Ch. 1

Enter the shell on application context and see how to get all defined URL routes,

```
flask shell

app.url_map
```

Useful information shows where Flask thinks our templates and static folders are:
```
>>> app.static_folder
'/home/topolo/PropD/Propulsion/T1000/FunctionalBoilerplate/FunctionalBoilerplate/static'
>>> app.template_folder
'templates'
>>> app.has_static_folder
True
```


# On Creating Views with Templates

cf. Ch. 3, Creating Views with Templates, Gaspar and Stouffer (2018)

## Jinja

**Jinja** is a templating language written in Python, 
**templating language** - it's a simple format designed to help automate creation of documents.
  - in any templating language, variables passed to the template replace predefined elements in the template
  - in Jinja, variable substitutions defined by {{ }}
`{{ }}` syntax is called **variable block**.

**control blocks** - `{% %}` - declare language functions, such as **loops** or `if` statements,
e.g. when `Post` instance, `post` gets passed to it, we get following Jinja code:

```
<h1>{{ post.title }}</h1>
```

producing

```
<h1>First Post</h1>
```

variables displayed in Jinja template can be any Python type or object as long as they can be converted into string via Python function `str()`, 
e.g. dictionary or list passed to template can have attributes displayed:

```
{{ your_dict['key'] }}
{{ your_list[0] }}
```

If you choose to combine Jinja and your JavaScript templates that are defined in your HTML files, then wrap JavaScript templates in the `raw` control block to tell Jinja to ignore them:

```
{% raw %}
<script id="template" type="text/x-handlebars-template">

{% endraw %}
```

### Jinja Filters

in Jinja, variables can be passed to built-in functions that modify the variables for display purposes.

These functions, called filters, are called in variable block with pipe character, `|`, e.g.

```
{{ variable | filter_name(*args) }}
```
or, if no arguments are passed to filter, parentheses can be omitted:
```
{{ variable | filter_name }}
```

## Creating our views

cf. pp. 58, Ch. 3, Gaspar and Stouffer (2018)

## WTForms basics

cf. pp. 69, Ch. 3, Gaspar and Stouffer (2018)

3 main parts of WTForms-
1. **forms**
2. **fields**
3. **validators**

validators are functions attached to fields that make sure data submitted in the form is within constraints.

The form is a class that contains fields and validators, and validates itself on a `POST` request.

The most common validators:
* `validators.Regexp(regex)`


`post.html`

pp. 73 Gaspar and Stouffer (2018)

`<form method="POST" action="{{ url_for('post', post_id=post.pid) }}">`
First, declare an HTML form section and make it submit (using `HTTP POST`) to our `post` Flask endpoint function with current post ID.

`form.hidden_tag()` adds anticross-site request forgetry measure automatically.


Then, when calling `field.label`, an HTML label will automatically be created for our input. This can be customized when we define our `WTForm FlaskForm` class; if not, WTForm will pretty print the field name.

# Class-based Views

pp. 76, Ch. 4, Gaspar and Stouffer




# Creating Controllers with Blueprints

cf. Ch. 4, Gaspar and Stouffer (2018)

**read Flasks' documentation carefully**, where variable attack methods are covered: `http:/​/flask.​pocoo.​org/docs/security/​`

```
        func.count(posts_tags_table.c.post_id).label('total')).join(
            posts_tags_table).group_by(Tag).order_by(
                desc('total')).limit(5).all()
```
Use SQLAlchemy `func` to return count on a group by query, we're able to order tags by the most used tags.



# References

Daniel Gaspar, Jack Stouffer. **Mastering Flask Web Development: Build enterprise-grade, scalable Python web applications**, 2nd Edition Kindle Edition. 2018.


Shalabh Aggarwal. **Flask Framework Cookbook**. Packt Publishing (November 21, 2014). ISBN-10 : 178398340X
ISBN-13 : 978-1783983407 


https://github.com/realpython/flask-boilerplate

https://exploreflask.com/en/latest/blueprints.html

