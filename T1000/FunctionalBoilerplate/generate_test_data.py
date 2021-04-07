from DatabaseSetup.test_data import (
    generate_posts,
    generate_tags,
    generate_users)


def generate_test_data(
    number_of_posts = 100,
    number_of_users = 10,
    number_of_tags = 5):

    generate_posts(
        number_of_posts,
        generate_users(number_of_users),
        generate_tags(number_of_tags))    


if __name__ == "__main__":

    generate_test_data()