cf. https://en.wikipedia.org/wiki/Create,_read,_update_and_delete
CRUD are 4 basic functions of persistent storage.

Operation, SQL, HTTP
Create - INSERT - POST (PUT is idempotent, meaning multiple identical requests have same effect as single request)
Read - SELECT - GET
Update - UPDATE - PUT / POST/ PATCH
Delete - DELETE - DELETE


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

Class name follows **Camelcase**.

#### Table

* representation of our table inside database

syntax:

`__tablename__ = 'some_table'`

make table names lowercase with underscore as convention

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

## Review of Clients, Servers

cf. https://classroom.udacity.com/courses/ud088/lessons/3593308716/concepts/36110092620923

Client-Server communication

Client - computer that wants information.
Server - computer that has information, that wants to share information

Client has to initiate communication.
Server stays listening.

Protocols are like the grammatical rules to make sure all computers communicate in same way.

Common protocols, TCP, IP (Internet Protocol), HTTP

TCP - transmission Control Protocol
broken in small packets to be sent.

UDP - user datagram protocol, good for streaming 

IP - similar to postal addresses, properly allows internet to get routed to all.
Looks up IP Addresses for Domain Name Service (DNS)

Ports

common port 80

Putting a colon means want to communicate on a specific port:
e.g. 66.249.95.1:8080

port 8080

localhost - when

special IP address 127.0.0.1
machine knows to look for this resource locally, and not go out to the internet.

Http - concept: Clients tell servers what they want by verbs

9 HTTP Verbs:

GET
Client telling 
Get safe method since read

POST
Client saying I want to modify some information

Status Code is 

200: Successful GET
201 
303

301 redirect
404: File not found


### Port forwarding

Port forwarding allows us to open pages in our browser from web server from our virtual machine as if they were being run locally.

More info about port forwarding:
https://www.vagrantup.com/docs/networking/forwarded_ports.html

# Message Flashing

`import flask import flask`

sessions a way a server can store information across multiple pages to create personal

Use `get_flashed_messages()` in template.

# RESTful API

cf. https://en.wikipedia.org/wiki/Representational_state_transfer
cf. https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586360923

Representational State Transfer - 
When HTTP is used, operations avaiable are GET, HEAD, POST, PUT, PATCH, DELETE, ...

When API is communicated over the internet, with rules of HTTP, called RESTful API

API (Application Programming Interfaces) allows external applications to use public info our apps want to share


