from . import (db_session, posts_tags_table, Comment, Post, Tag, User)

from collections import OrderedDict
from flask import (
    Blueprint,
    render_template,
    request)
from flask.views import (MethodView, View)
from sqlalchemy import (and_, or_)


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


@examples_bp.route('/htmlDataTables')
def html_datatables():
    return render_template(
        'pages/dataTablesExamples.html',
        title="DataTables examples")


def sqlalchemy_table_query_to_dict(
        query_results,
        target_mapper_class=None):

    if not isinstance(query_results, list):
        return None, None

    if target_mapper_class != None:

        table_columns = list(target_mapper_class.__table__.columns.keys())

    else:

        if (len(query_results) > 0):

            try:

                table_columns = list(query_results[0].__table__.columns.keys())

            except AttributeError as err:
                print("\n Attribute Error found:", err)
                print("\n dir of query_results[0]: ", dir(query_results[0]))

                table_columns = list(query_results[0].keys())


        else:

          return None

    def to_row_dict(query_result, input_table_columns):
        # cf. https://stackoverflow.com/questions/1167398/python-access-class-property-from-string

        return OrderedDict(
            [(col, getattr(query_result, col)) for col in input_table_columns])

    return (
        table_columns,
        [
            to_row_dict(query_result, table_columns)
            for query_result in query_results])


# cf. https://flask.palletsprojects.com/en/2.0.x/quickstart/#accessing-request-data

"""
@ref pp. 168, HTML5: Search Input, Forms, Duckett, HTML and CSS.

@details For a 'GET' method, the input from the search input form returns as
part of the URL.
"""
@examples_bp.route('/htmlSearchInput', methods=['GET', 'POST'])
def html_search_input_user():

    search_input = None

    user_columns = None
    user_rows = None
    post_columns = None
    post_rows = None

    if request.method =='GET':

        form_data = request.form.to_dict(flat=False)

        print("\n form_data: ", form_data)

        #search_input = request.form['search_user_submitted']

    if request.method == 'POST':

        form_data = request.form.to_dict(flat=False)

        print("\n form_data from POST: ", form_data)

        if form_data:

            search_input = request.form['search_user']       

            # Need to add % for wildcard like search in SQLAlchemy.

            modified_search_input = search_input + '%'

            # cf. https://stackoverflow.com/questions/2128505/difference-between-filter-and-filter-by-in-sqlalchemy 
            # filter vs. filter_by in SqlAlchemy
            query_results = User.query.filter(
                getattr(User, 'username').like(modified_search_input)).all()

            print("\n query_results in POST :", query_results)
            print(type(query_results))

            query_results_with_posts = Post.query.join(User).filter(
                getattr(User, 'username').like(modified_search_input)).all()

            print(
                "\n query_results with posts in POST :",
                query_results_with_posts)
            print(type(query_results_with_posts))
            print(type(query_results_with_posts[0]))

            user_columns, user_rows = sqlalchemy_table_query_to_dict(
                query_results,
                User)

            post_columns, post_rows = sqlalchemy_table_query_to_dict(
                query_results_with_posts)


    return render_template(
        'requests/example_search.html',
        title="HTML5 Search Input - user",
        search_input=search_input,
        user_columns=user_columns,
        user_rows=user_rows,
        post_columns=post_columns,
        post_rows=post_rows)


class UserSearch(MethodView):

    def __init__(self):
        self.title = "User Search example Method View"
        self.get_html_template = 'requests/example_search.html'
        self.post_html_template = 'requests/example_post_query_results.html'

    def get_user_search(self, search_input, column_to_search='username'):

        # Need to add % for wildcard like search in SQLAlchemy

        modified_search_input = search_input + '%'

        user_query = User.query.filter(
            getattr(User, 'username').like(modified_search_input)).all()

        comment_query = Comment.query.join(Post).join(User).filter(
            getattr(User, 'username').like(modified_search_input)).all()

        user_columns, user_rows = sqlalchemy_table_query_to_dict(
            user_query,
            User)

        comment_columns, comment_rows = sqlalchemy_table_query_to_dict(
            comment_query)

        return (user_columns, user_rows, comment_columns, comment_rows)

    def get(self):

        return render_template(
            self.get_html_template,
            title=self.title,
            search_input=None,
            user_columns=None,
            user_rows=None,
            post_columns=None,
            post_rows=None)

# Need argument 'user_search' for this blueprint as a variable in jinja.
user_search = UserSearch.as_view('user_search')

examples_bp.add_url_rule(
    '/userSearch',
    view_func=user_search,
    methods=['GET',])

"""
@examples_bp.route('/htmlGetForm', methods=['GET'])
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


class AllUsers(View):

    def __init__(self, html_template='pages/allUsers.html'):
        self.html_template = html_template
        self.title = "All Users example view"


    def get_user_objects(self):

        print("\n ----- get_user_object has run ----- \n")

        # type, Python list.
        user_object = User.query.all()
        print("\n dir of User: ", dir(User))
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

def get_all_for_select_and_clean(TargetTable, display_columns, value_columns):
    all_results = TargetTable.query.all()

    def parse_row(row):
        key = getattr(row, display_columns[0])
        value = getattr(row, value_columns[0])
        return (key, value)

    return [parse_row(row) for row in all_results]

def get_all_for_display_and_clean(TargetTable):
    all_results = TargetTable.query.all()

    return sqlalchemy_table_query_to_dict(all_results, TargetTable)


class MultipleSelectInteraction(MethodView):

    relative_location = '/multipleSelectExamples'

    py_jinja_name = 'multiple_select_interaction'

    title = 'Multiple Select Interaction'

    def __init__(self):
        self.get_multiple_select_template = 'pages/multipleSelectExamples.html'

    def get_all_users_and_clean(self):

        # These come in as external inputs.
        display_columns = ['username',]
        data_columns = ['uid',]

        return get_all_for_select_and_clean(
            User,
            display_columns,
            data_columns)

    def query_posts_from_user_id(self, user_id):
        query_results = (db_session.query(
            User.username,
            User.uid,
            Post.title,
            Comment.name).
                filter(User.uid==user_id).
                filter(User.uid==Post.user_id).
                filter(Post.pid==Comment.post_id)).all()

        if (len(query_results) == 0):
            return None

        return sqlalchemy_table_query_to_dict(query_results)

    def query_tags_from_user_id(self, user_id):
        query_results = (db_session.query(
            User.username,
            User.uid,
            Post.title,
            Post.pid,
            Post.publish_date,
            Tag.tid).
                filter(User.uid==user_id).
                filter(User.uid==Post.user_id).
                filter(Post.pid==posts_tags_table.columns.post_id).
                filter(posts_tags_table.columns.tag_id==Tag.tid)).all()

        if (len(query_results) == 0):
            return None

        return sqlalchemy_table_query_to_dict(query_results)

    def get(self):

        all_users_list = self.get_all_users_and_clean()

        return render_template(
            self.get_multiple_select_template,
            title=MultipleSelectInteraction.title,
            all_users_list=all_users_list,
            single_user_select_id=None,
            user_post_comment=None,
            user_post_tag=None)

    def post(self):

        all_users_list = self.get_all_users_and_clean()

        # request has no attribute POST
        #print("\n request.POST : ", request.POST)
        #print(dir(request.POST))

        # Python string type.
        selected_user_id = request.form['single_user_select']
        print("\n type selected_user_id:", type(selected_user_id))

        selected_user_id = int(selected_user_id)

        user_post_comment_data = self.query_posts_from_user_id(
            selected_user_id)
        print("\n user post comment: ", user_post_comment_data)

        user_post_tag_data = self.query_tags_from_user_id(
            selected_user_id)

        print("\n dir of post_tags_table: ", dir(posts_tags_table))
        print(type(posts_tags_table))

        return render_template(
            self.get_multiple_select_template,
            title=MultipleSelectInteraction.title,
            all_users_list=all_users_list,
            single_user_select_id=request.form['single_user_select'],
            user_post_comment=user_post_comment_data,
            user_post_tag=user_post_tag_data)


examples_bp.add_url_rule(
    MultipleSelectInteraction.relative_location,
    view_func=MultipleSelectInteraction.as_view(
        MultipleSelectInteraction.py_jinja_name),
    methods=['GET', 'POST'])


class MultipleMultipleSelect(MethodView):
    relative_location = '/multipleMultipleSelectExamples'
    py_jinja_name = 'multiple_multiple_select'
    title = 'Multiple Multiple Select'
    http_template = 'requests/multipleMultipleSelectExamples.html'

    def __init__(self):
        super(MultipleMultipleSelect, self).__init__()

        # These come in as external inputs.
        display_columns = ['username',]
        data_columns = ['uid',]

        self.all_users_list = get_all_for_select_and_clean(
            User,
            display_columns,
            data_columns)

    def get(self):

        return render_template(
            MultipleMultipleSelect.http_template,
            title=MultipleMultipleSelect.title,
            all_users_list=self.all_users_list,
            selected_user_ids=None,
            searched_posts=None)

    def query_multiple_users(self, selected_user_ids):
        query_results = (db_session.query(
            User.username,
            User.uid,
            Post.title,
            Post.pid,
            Post.user_id).
                filter(
                    or_((User.uid == s_uid)
                        for s_uid in selected_user_ids)).
                filter(User.uid == Post.user_id)
                ).all()
        if (len(query_results) == 0):
            return None

        return sqlalchemy_table_query_to_dict(query_results)

    def post(self):
        selected_user_ids = None
        searched_posts = None

        form_data = request.form.to_dict(flat=False)
        print("\n form data: ", form_data)

        if (request.form['multiple_user_select'] != 'NoneSelected'):

            raw_selected_user_ids = request.form.to_dict(flat=False)[
                'multiple_user_select']

            raw_selected_user_ids = [
                int(x) for x in raw_selected_user_ids
                if x != 'NoneSelected']

            if (len(raw_selected_user_ids) > 0):
                selected_user_ids = raw_selected_user_ids

            print("\n selected user ids: ", selected_user_ids)

            selected_user_ids = self.query_multiple_users(selected_user_ids)

            #selected_user_ids = [
            #    int(x) for x in request.form['multiple_user_select']]

        if (form_data['post_title_search'][0] != ''):
            print("\n Nonempty post_title_search \n")

            modified_search_input = form_data['post_title_search'][0] + '%'

            print("\n modified_search_input: ", modified_search_input)

            query_results = Post.query.filter(
                getattr(Post, 'title').like(modified_search_input)).all()

            searched_posts = sqlalchemy_table_query_to_dict(
                query_results,
                Post)

            print("\n searched_posts : ", searched_posts)


        else:
            print("\n EMPTY post_title_search \n")


        return render_template(
            MultipleMultipleSelect.http_template,
            title=MultipleMultipleSelect.title,
            all_users_list=self.all_users_list,
            selected_user_ids=selected_user_ids,
            searched_posts=searched_posts)


examples_bp.add_url_rule(
    MultipleMultipleSelect.relative_location,
    view_func=MultipleMultipleSelect.as_view(
        MultipleMultipleSelect.py_jinja_name),
    methods=['GET', 'POST'])


class AddUser(MethodView):
    relative_location = '/addUser'
    py_jinja_name = 'add_user'
    title = 'Add New User'
    http_template = 'requests/add_user.html'

    def __init__(self):
        super(AddUser, self).__init__()

        self.all_users_list = get_all_for_display_and_clean(User)

    def get(self):

        return render_template(
            AddUser.http_template,
            title=AddUser.title,
            all_users_list=self.all_users_list)

    def post(self):
        form_data = request.form.to_dict(flat=False)
        print("\n form data: ", form_data)

        if (form_data['add_new_username']):
            print("\n Adding new username :", form_data['add_new_username'])
            new_user = User()
            new_user.username = form_data['add_new_username'][0]
            new_user.password = form_data['add_new_password'][0]

            print(form_data['add_new_username'])
            print(form_data['add_new_password'])

            db_session.add(new_user)

            db_session.commit()


        return render_template(
            AddUser.http_template,
            title=AddUser.title,
            all_users_list=self.all_users_list)


examples_bp.add_url_rule(
    AddUser.relative_location,
    view_func=AddUser.as_view(AddUser.py_jinja_name),
    methods=['GET', 'POST'])

