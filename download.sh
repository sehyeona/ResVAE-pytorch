FILE=$1

if  [ $FILE == "top_img" ]; then
    URL=https://www.dropbox.com/s/zk6jxt4xce79t0b/topimg.zip?dl=0
    ZIP_FILE=./data/top_img.zip
    mkdir -p ./data/top_img
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/top_img
    rm $ZIP_FILE

elif  [ $FILE == "bottom_img" ]; then
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=./data/bottom_img.zip
    mkdir -p ./data/bottom_img
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/bottom_img
    rm $ZIP_FILE

else
    echo "Available arguments are top_img/bottom_img."
    exit 1

fi
