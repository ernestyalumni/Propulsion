from flask import (
    Blueprint,
    render_template)


about_bp = Blueprint(
    'about_bp',
    __name__,
    static_folder='../static',
    template_folder='../templates'
    )

@about_bp.route('/about')
def about():
    return render_template(
        'pages/placeholder.about.html',
        title="Functional Boilerplate about",
        subtitle="About Functional structure for Flask")