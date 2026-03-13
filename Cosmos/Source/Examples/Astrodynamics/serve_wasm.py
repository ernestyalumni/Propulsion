# save as serve_wasm.py  (anywhere you like, e.g. the Examples folder)
import http.server, socketserver, mimetypes, os

# -------------------------------- -------------------------------- ------                                                                                                                                  
# 1️⃣  Extend SimpleHTTPRequestHandler to force the right MIME for .wasm                                                                                                                                     
# -------------------------------- -------------------------------- ------                                                                                                                                  

class WasmHandler(http.server.SimpleHT TPRequestHandler):
    # Called whenever a file is about to be sent                                                                                                                                                            
    def guess_type(self, path):
        base_type, _ = super().guess_type(path)                                                                                                                                                             
        # 👉 Force the official WASM MIME type                                                                                                                                                              
        if path.endswith('.wasm'):                                                                                                                                                                          
            return 'application/wasm'                                                                                                                                                                       
        return base_type                                                                                                                                                                                    
                                                                                                                                                                                                            
# -------------------------------- -------------------------------- ------                                                                                                                                  
# 2️⃣  Change the working directory to where the demo lives                                                                                                                                                  
# -------------------------------- -------------------------------- ------                                                                                                                                  
WEB_ROOT = '/media/propdev/Expansion/openclaw/.openclaw/workspace/repos/Propulsion/Cosmos/Source/Examples/Astrodynamics'                                                                                    
os.chdir(WEB_ROOT)                                                                                                                                                                                          
                                                                                                                                                                                                            
# -------------------------------- -------------------------------- ------                                                                                                                                  
# 3️⃣  Start the server on port 8000 (bind to localhost only)                                                                                                                                                
# -------------------------------- -------------------------------- ------                                                                                                                                  
PORT = 8000                                                                                                                                                                                                 
with socketserver.TCPServer(('127.0.0 .1', PORT), WasmHandler) as httpd:                                                                                                                                    
    print(f'Serving {WEB_ROOT} at http://127.0.0.1:{PORT}')                                                                                                                                                 
    print(' → Open this URL in Chrome: http://127.0.0.1:8000/NumerovOrbitDemoWasm.html')                                                                                                                    
    httpd.serve_forever()