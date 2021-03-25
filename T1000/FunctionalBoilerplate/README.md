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


# References

Daniel Gaspar, Jack Stouffer. **Mastering Flask Web Development: Build enterprise-grade, scalable Python web applications**, 2nd Edition Kindle Edition. 2018.