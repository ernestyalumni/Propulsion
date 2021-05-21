from flask import (
    Blueprint,
    render_template)


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