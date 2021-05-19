IntroToHTML5.md

[01.03 - How it Works: The "Magic" of Page Requests](https://www.coursera.org/learn/html/lecture/5syEt/01-03-how-it-works-the-magic-of-page-requests)

Type address into URL bar, 

### Client/Server Relationship

* Servers
	* Machines that hold shared resources
	* Always connected to network

* Clients
	* Machines for personal use (laptops, phones, etc.)

### Networks

* The internet
	* LAN (Local Area Network, shares only one server, for example share only one printer)
	* WAN (Wide Area Network, share servers across multiple buildings)

### Request/Response Cycle

* This is what happens when your computer (the client) requests a page and a server responds with the appropriate files
(when you type URL address, you're requesting a webpage, and server has to respond with file)

### Uniform Resource Locator (URL)

* URL - 3 parts
	* protocol - how to connect
	* domain - the server*
	* (optional) document - the specific file needed
		* Most pages are made up of multiple files

one URL, requesting lots of files typically.

### Protocols

* HTTP 
* HTTPS
* FTP

(with FTP could be anytype of file)

### Domain Names

* Identifies the entity you want to connect to 
	* `umich.edu`, `google.com`, `wikipedia.org`
ICANN determines which type of organizations qualify for what type of domain.

* Each has different top-level domain
	* Determined by Internet Corporation for Assigned Names and Numbers (ICAAN)
	* `https://www.icann.org/resources/pages/tlds-2012-02-25-en`


(*Domain name is mapped to an address*)

### IP Addresses

* Internet Protocol Version 6 (IPv6) is the communication protocol that identifies computers on networks.

* Every computer has a unique IP address

xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx

8 groups of 4 = 32, = 2^5

where x can have 16 different values: 2^4 = 16

* Can represent over 300 trillion unique combinations (2^128)!

### The Domain Name Server (DNS)

* Luckily, you don't need to remember IP address of a domain.
* The DNS will lookup IP address based on URL you type in.

### Document

* URLs can specify a specific document
	* `http://www.intro-webdesign.com/ ` **contact.html**
	* `http://www.intro-webdesign.com/` **Ashtabula/harbor.html**

* If no document is specified, default document is returned.
	* Convention is **`index.html`**

### The Request

* Once IP address is determined, browser creates an HTTP request.

* Lots of hidden information in this request
	* header, cookies, form data, etc.

(**The important thing is server returns files, not "web pages"**)

### The Response

* The server returns files, not "web pages"
	* It's up to browser to decide what to do with those files

* If server can't fulfill request, it'll send back files with error codes: 404, 500, etc.

### Review (How it Works: The "Magic" of Page Requests)

* A URL has 3 parts
* Request/Response cycle typically requires multiple rounds of communication between client and server

https://www.techradar.com/best/browser

http://html5test.com/index.html

