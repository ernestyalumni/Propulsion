"""
@file webserver.py

@ref https://classroom.udacity.com/courses/ud088/lessons/3593308716/concepts/36082987050923
"""

# Python 2
# from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

from http.server import BaseHTTPRequestHandler, HTTPServer

# Common Gateway Interface scripts 
import cgi

# When writing a new script, consider adding these lines:
# cf. https://docs.python.org/3/library/cgi.html
# This activates a special exception handler that'll display detailed reports
# in the Web browser if any errors occur.
#import cgitb
#cgitb.enable()

# class http.server.HTTPServer(server_address, RequestHandlerClass)
# builds on TCPServer class by storing server address as instance variables
# named server_name, and server_port. Server accessible by handler, typically
# through handler's server instance variable
# server_address is a tuple containing host and port number.

# https://docs.python.org/3/library/http.server.html#module-http.server

# BaseHTTPRequestHandler(request, client_address, server)
# handles HTTP requests that arrive at server; by itself, it can't respond to
# any actual HTTP requests; it must be subclassed to handle each request method
# (e.g. GET or POST).
# BaseHTTPRequestHandler provides number of class and instance variables, and
# methods for use by subclasses.
#
# Handler will parse request and headers, then call method specific to the
# request type. The method name is constructed from the request.


# Handler code

class WebServerHandler(BaseHTTPRequestHandler):
    """
    @class WebServerHandler

    @brief Handles requests, handling what code to execute, based on HTTP code
    sent to it.
    """
    def do_GET(self):
        """
        @details The request is mapped to a local file by interpreting the
        request as path relative to current working directory.

        If request mapped to a directory, directory checked for file named
        index.html or index.htm (in that order). If found, file's content are
        returned.
        """

        try:

            # BaseHTTPRequestHandler provides variable path that is path of
            # address given.

            # self.path path Contains the request path
            if self.path.endswith("/hello"):

                # send_response(code, message=None)
                # Adds response header to headers buffer and logs accepted
                # requested. HTTP response line written to internal buffer,
                # followed by Server and Date headers.
                # Values for these 2 headers are picked up from
                # version_string() and date_time_string() methods, respectively
                # If server doesn't intend to send any other headers using
                # send_header() method, then send_response() should be followed
                # by end_headers() call.
                self.send_response(200)

                self.send_header('Content-type', 'text/html')

                self.end_headers()

                output = ""
                output += "<html><body><h1>Hello! Bonjour!</h1></body></html>"

                # Contains output stream for writing response back to client.
                # Proper adherence to HTTP protocol must be used when writing
                # to this stream in order to achieve successful interoperation
                # with HTTP clients.

                # https://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3
                self.wfile.write(output.encode("utf-8"))

                # Useful for debugging.
                print(output)
                return

            # cf. https://classroom.udacity.com/courses/ud088/lessons/3593308716/concepts/36082987070923
            # Responding to Multiple GET Requests

            if self.path.endswith("/hola"):

                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()

                output = ""
                output += "<html><body><h1>&#161Hola  <a href = '/hello' >Back to Hello</a></h1></body></html>"

                self.wfile.write(output.encode("utf-8"))

                print(output)
                return

        def do_POST(self):

            try:

                self.


            except:


        except IOError:

            self.send_error(404, "File Not Found %s" % self.path)


def create_http_server(port, WebServerHandler, server_name=''):
    return HTTPServer((server_name, port), WebServerHandler)

# Instantiates server and what port to listen on.

def main():

    try:

        port = 8080
        server = create_http_server(port, WebServerHandler)

        print("Web server running on port %s" % port)

        # Until I keep it constantly listening, until keyboard interrupt.
        server.serve_forever()

    # KeyboardInterrupt is a built-in exception triggered when user holds
    # Ctrl-C
    except KeyboardInterrupt:

        print("^C entered, stopping web server...")
        server.socket.close()

# To immediately run main function, as the Python interpreter executes this
# script.

if __name__ == '__main__':

    print("\n Running main() function: \n")

    main()
