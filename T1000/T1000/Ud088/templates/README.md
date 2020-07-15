<!-- menu.html
  https://github.com/udacity/Full-Stack-Foundations/blob/master/Lesson-3/07_Menu-Template-Quiz/menu.html 
  Lesson-3 / 07_Menu-Template-Quiz / menu.html -->

<!-- Flask or specifically render_template command will look for templates in a folder called templates.

HTML escaping is putting Python code in HTML; Flask does this automatically -->

<html>

<body>

<h1>{{restaurant.name}}</h1>

<!-- https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586180923 -->
<!-- brace percent sign {% is syntax for logical code brace logical code you want to execute -->
{% for i in items %}

<div>

<!-- {{ printed code }} -->
<p>{{i.name}}</p>

<p>{{i.description}}</p>

<p>{{i.price}}</p>

<!-- https://github.com/udacity/Full-Stack-Foundations/blob/master/Lesson-3/09_url_for-quiz/menu.html
Lesson 3 09_url_for-quiz -->
<!-- URL Building 
cf. https://flask.palletsprojects.com/en/1.1.x/api/
flask.url_for(endpoint, **values)

Generates a URL to given endpoint with method provided.
Variable arguments that are unknown to target endpoint are appended to generated URL as query arguments.
If value of query argument is None, whole pair is skipped.
In case blueprints are active, you can shortcut references to same blueprint by prefixing local endpoint with dot (.)

This will reference index function local to current blueprint:
url_for('.index')

url_for(_,_,_, ...)
function and keyword arguments
https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586210923
url_for('edit_menu_item', restaurant_id = restaurant_id = restaurant.id, menu_id = i.id) }}

Can change URLs dynamically.

-->


<!--
<a href = "{{url_for('edit_menu_item', restaurant_id = restaurant.id, menu_id = i.id) }}">Edit</a>

</br>

<a href = "{{url_for('delete_menu_item', restaurant_id = restaurant.id, menu_id = i.id) }}">Delete</a>
-->
</div>


{% endfor %}

</body>

</html>


<!-- edit_menu_item.html -->

<html>
<body>

<!--
cf. https://www.w3schools.com/tags/att_form_action.asp
On submit, send the form-data to a file named or the html named in action

Syntax:
<form action="URL">

URL - where to send form-data when form is submitted.
Possible values:
* absolute url
* relative url

-->

<form action= "{{url_for('edit_menu_item', restaurant_id = restaurant_id, menu_id=item.id)}}" method = 'POST'>

<p>Name:</p>

<!-- placeholder allows for seeing what item is being edited -->
<input type = 'text' size='30' name = 'name' placeholder = '{{item.name}}'>

<!-- add a button called Edit -->
<input type='submit' value='Edit'>
</form>

<!-- create a link to cancel -->
<a href= '{{url_for('restaurant_menu', restaurant_id= item.restaurant_id)}}'>Cancel</a>

</body>
</html>