

def run_examples():

    from .create_application import create_app
    from flask import url_for


    app_instance = create_app()

    print("\n Welcome to Flask Shell examples\n")

    # Gaspar, pp. Getting Started

    print(app.url_map)
    print(app.static_folder)
    print(app.template_folder)

    #url_for_post_string = url_for('/posts')

    #print(url_for_post_string)

    return app_instance


if __name__ == "__main__":

    app_instance = run_examples()
