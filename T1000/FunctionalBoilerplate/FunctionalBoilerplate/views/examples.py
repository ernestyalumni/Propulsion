from . import User

from flask import (
    Blueprint,
    render_template)
from flask.views import (MethodView, View)

from collections import OrderedDict


class ExamplesBlueprintConstants:
    url_prefix = "/examples"


examples_bp = Blueprint(
    'examples_bp',
    __name__,
    url_prefix=ExamplesBlueprintConstants.url_prefix)


@examples_bp.route('/')
@examples_bp.route('/htmlPage')
def html_page():
    return render_template(
        'pages/frontPage.html',
        title="HTML front page",
        subtitle="Demonstrate HTML")


@examples_bp.route('/htmlPageInherited')
def html_page_inherited():
    return render_template(
        'pages/frontPageInherited.html',
        title="HTML front page inherited",
        subtitle="Demonstrate HTML with inheritance")

# cf. https://flask.palletsprojects.com/en/2.0.x/quickstart/#accessing-request-data

"""
@examples_bp.route('/htmlGetForm')
def html_get_form():
    response = None

    if request.method == 'GET':
"""


"""
@brief Example demonstrating forms, and class-based views.

@ref pp. 151, Forms, Duckett, HTML and CSS.
"""

class FormView(View):

    methods = ['GET']

    def __init__(self, html_template='pages/formExample.html'):
        self.html_template = html_template

        self.title = "FormView example"
        self.arbitrary_variable = "arbitraryVariable"

        super(FormView, self).__init__()

    def render_template(self, context):
        return render_template(self.html_template, **context)

    def dispatch_request(self):

        context = {
            'title': self.title,
            'arbitrary_variable': self.arbitrary_variable
        }

        return self.render_template(context)


examples_bp.add_url_rule(
    '/htmlForm',
    view_func=FormView.as_view('form_example'))


def sqlalchemy_table_query_to_dict(
        query_results,
        target_mapper_class=None):

    if not isinstance(query_results, list):
        return None

    if target_mapper_class != None:

        table_columns = list(target_mapper_class.__table__.columns.keys())

    else:
        table_columns = list(query_result[0].__table__.columns.keys())

    def to_row_dict(query_result, input_table_columns):
        # cf. https://stackoverflow.com/questions/1167398/python-access-class-property-from-string

        return OrderedDict(
            [(col, getattr(query_result, col)) for col in input_table_columns])

    return (
        table_columns,
        [
            to_row_dict(query_result, table_columns)
            for query_result in query_results])


class AllUsers(View):

    def __init__(self, html_template='pages/allUsers.html'):
        self.html_template = html_template
        self.title = "All Users example view"


    def get_user_objects(self):


        print("\n ----- get_user_object has run ----- \n")

        # type, Python list.
        user_object = User.query.all()
        print("\n dir of User: ", dir(User))
        print("\n dir of user_object: ", dir(user_object))
        print("\n type of user_object: ", type(user_object))
        print(len(user_object))
        # FunctionalBoilerplate.Model.user.User        
        print(type(user_object[0]))
        print(dir(user_object[0]))

        user_columns = User.__table__.columns.keys()
        print("\n dir of user_columns: ", dir(user_columns))
        print("\n type of user_columns: ", type(user_columns))
        print(len(user_columns))
        # (Python) string
        print(type(user_columns[0]))
        print(dir(user_columns[0]))

        return sqlalchemy_table_query_to_dict(user_object, User)

    def render_template(self, context):
        return render_template(self.html_template, **context)


    def dispatch_request(self):

        user_columns, user_objects = self.get_user_objects()

        print("\n user_columns: ", user_columns)
        print("\n user_objects: ", user_objects)

        context = {
            'user_columns': user_columns,
            'user_objects': user_objects,
            'title': self.title}

        return self.render_template(context)


examples_bp.add_url_rule(
    '/allUsers',
    # 'all_users' defines how to name this URL for Python flask, when doing
    # examples_bp.all_users
    view_func=AllUsers.as_view('all_users'))
