
"""
@ref Ch. 4, pp. 78 Gaspar and Stouffer (2018), "Class-based views"

"""
from . import (
    Comment,
    Post,
    User)


from flask import (
    Blueprint,
    render_template)
from flask.views import (MethodView, View)


class ClassBasedViewsConstants:
    url_prefix = "/class_based_views/"


class_based_views_bp = Blueprint(
    'class_based_views_bp',
    __name__,
    url_prefix=ClassBasedViewsConstants.url_prefix)


class GenericListView(View):
    """
    @ref https://github.com/PacktPublishing/Mastering-Flask-Web-Development-Second-Edition/blob/master/Chapter04/main.py
    """

    def __init__(self, model, list_template='posts/generic_list.html'):
        self.model = model
        self.list_template = list_template
        self.columns = self.model.__mapper__.columns.keys()
        # Call super python3 style
        super(GenericListView, self).__init__()

    def render_template(self, context):
        return render_template(self.list_template, **context)

    def get_objects(self):
        return self.model.query.all()

    def dispatch_request(self):
        """
        @ref https://flask.palletsprojects.com/en/1.1.x/api/?highlight=view#flask.views.View

        @details Subclass has to implement dispatch_request() which is called
        with view arguments from URL routing system

        Subclasses have to override this method to implement the actual view
        function code. This method is called with all arguments from URL rule.
        """

        context = {
            'objects': self.get_objects(),
            'columns': self.columns}
        return self.render_template(context)


"""
@ref https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.add_url_rule

add_url_rule(rule, endpoint=None, view_func=None, provide_automatic_options,
None,**options)

Connects a URL rule, Works exactly like the route() decorator. If a view_func
is provided, it'll be registered with the endpoint.

@app.route('/')
def index():
    pass

equivalent to

def index():
    pass
app.add_url_rule('/', 'index', index)

Parameters
* rule - the URL rule as string
* endpoint - the enpoint for the registered URL rule. Flask itself assumes the
name of the view function as endpoint
* view_func - function to call when serving a request to the provided endpoint
* provide_automatic_options - controls whether OPTIONS method should be added
automatically. This can also be controlled by setting the
view_func.provide_automatic_options = False before adding the rule.
* options - the options to be forwarded to underlying Rule object. A change to
Werkzeug is handling of method options. methods is a list of methods this rule
should be limited to (GET, POST etc.).
By default a rule just listens for GET
"""

class_based_views_bp.add_url_rule(
    '/generic_posts',
    view_func=GenericListView.as_view('generic_posts', model=Post))

class_based_views_bp.add_url_rule(
    '/generic_users',
    view_func=GenericListView.as_view('generic_users', model=User))

class_based_views_bp.add_url_rule(
    '/generic_comments',
    view_func=GenericListView.as_view('generic_comments', model=Comment))