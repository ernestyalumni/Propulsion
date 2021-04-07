from flask.views import View

class GenericListView(View):

    def __init__(self, model, list_template='generic_list.html'):

        self.model = model
        self.list_template = list_template
        self.columns = self.model.__mapp__.columns.keys()
        # Call super python3 stype
        super(GenericListView, self).__init__()

    def render_template(self, context):
        return render_template(self.list_template, **context)

    def get_objects(self):
        return self.model.query.all()

    def dispatch_request(self):

        context = {
            'objects': self.get_objects(),
            'columns': self.columns}

        return self.render_template(context)


def add_generic_posts_url_rule(app):
    app.add_url_rule(
        '/generic_posts',
        view_func=GenericListView.as_view(
            'generic_posts',
            model=Post))

    
