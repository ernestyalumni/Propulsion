from flask_wtf import Form
from wtforms import TextField, PasswordField
from wtforms.validators import DataRequired, EqualTo, Length

class LoginForm(Form):
    name = TextField('Username', [DataRequired()])
    password = PasswordField('Password', [DataRequired()])


class RegisterForm(Form):
    name = TextField(
        'Username',
        validators=[DataRequired(), Length(min=6, max=25)])

    password = PasswordField(
        'Password',
        validators=[DataRequired(), Length(min=6, max=40)])

    confirm = PasswordField(
        'Repeat Password',
        [DataRequired(),
        EqualTo('password', message='Passwords must match')])


class ForgotForm(Form):
    username = TextField(
        'Username', validators=[DataRequired(), Length(min=6, max=40)])