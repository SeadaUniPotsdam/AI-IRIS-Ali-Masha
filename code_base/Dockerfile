FROM busybox

# Set working directory
WORKDIR /tmp/code_base

# Copy script and README
COPY iris.py codeBase_Iris.py
COPY README.md /tmp/knowladge_base

# Install required Python libraries (BusyBox does not include Python, so this will need adjustments)
RUN wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz \
    && tar -xzf Python-3.13.1.tgz \
    && cd Python-3.13.1 \
    && ./configure \
    && make \
    && make install \
    && pip install --no-cache-dir numpy pandas tensorflow matplotlib

# Command to run the script (optional)
CMD ["python", "codeBase_Iris.py"]
