
# Current time
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME"
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR