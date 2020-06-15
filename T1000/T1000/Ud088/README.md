cf. [Creating a Database - Configuration](https://classroom.udacity.com/courses/ud088/lessons/3621198668/concepts/36123887280923)

Creating a database with SQLAlchemy

- Configuration
- Class
- Table
- Mapper

### Configuration

* generally shouldn't change from project to project

#### At beginning of file

* imports all modules needed
* creates instance of declarative base

#### At end of file

* creates (or connects) database and adds tables and columns

### Class

* representation of table as a python class

* extends the Base class

* nested inside will be table and mapper code

Class name follows Camelcase.

#### Table

* representation of our table inside database

syntax:

`__tablename__ = 'some_table'`


#### Mapper

mapper code maps python objects to columns in our database.

syntax
```
columnName = Column(attributes, ...)
```

## Update 

cf. [CRUD Review](https://classroom.udacity.com/courses/ud088/lessons/3621198668/concepts/36300689480923)

To update an existing entry in our database,

1. Find Entry
2. Reset value(s)
3. Add to session
4. Execute `session.commit()`


# Lesson 2: Making a Web Server

## REview of Clients, Servers

Client-Server communication

Client - computer that wants information.