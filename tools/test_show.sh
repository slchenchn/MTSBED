set -x

PTH=$1
SHOW_DIR=$(dirname "${PTH}")/show

# sleep 30s
python tools/test.py ${PTH} --show-dir ${SHOW_DIR} --opacity 1.0