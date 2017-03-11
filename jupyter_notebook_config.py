# Copyright (c) Jupyter Development Team.
from jupyter_core.paths import jupyter_data_dir
import subprocess
import os
import errno
import stat

PEM_FILE = os.path.join(jupyter_data_dir(), 'notebook.pem')

c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.port = 3006
c.NotebookApp.open_browser = False

if not os.path.isfile(PEM_FILE):
    # Ensure PEM_FILE directory exists
    dir_name = os.path.dirname(PEM_FILE)
    try:
        os.makedirs(dir_name)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dir_name):
            pass
        else: 
            raise
    # Generate a certificate if one doesn't exist on disk
    subprocess.check_call(['openssl', 'req', '-new', 
    '-newkey', 'rsa:2048', '-days', '365', '-nodes', '-x509',
    '-subj', '/C=XX/ST=XX/L=XX/O=generated/CN=generated',
    '-keyout', PEM_FILE, '-out', PEM_FILE])
    # Restrict access to PEM_FILE
    os.chmod(PEM_FILE, stat.S_IRUSR | stat.S_IWUSR)
c.NotebookApp.certfile = PEM_FILE

c.NotebookApp.password = 'sha1:22fa6ddb4b29:bcea37260ead285111122de31b21de9997e588eb'
