from ..forms.forms import (
    ForgotForm,
    LoginForm,
    RegisterForm)

from flask import (
    Blueprint,
    render_template,
    request)


form_bp = Blueprint(
    'form_bp',
    __name__,
    static_folder='../static',
    template_folder='../templates'
    )

@form_bp.route('/login')
def login():
    # cf. https://flask.palletsprojects.com/en/1.1.x/api/#flask.render_template
    # flask.render_template(template_name_or_list, **context)
    # * context - the variables that should be available in the context of the
    # template
    form = LoginForm(request.form)
    return render_template(
        'forms/login.html',
        form=form)

@form_bp.route('/register')
def register():
    form = RegisterForm(request.form)
    return render_template('forms/register.html', form=form)


@form_bp.route('/forgot')
def forgot():
    form = ForgotForm(request.form)
    return render_template('forms/forgot.html', form=form)