wget http://cvcl.mit.edu/scenedatabase/coast.zip
wget http://cvcl.mit.edu/scenedatabase/opencountry.zip
wget http://cvcl.mit.edu/scenedatabase/forest.zip
wget http://cvcl.mit.edu/scenedatabase/mountain.zip

mkdir scenedatabase
mkdir scenedatabase/images

unzip coast.zip -d scenedatabase/images
unzip opencountry.zip -d scenedatabase/images
unzip forest.zip -d scenedatabase/images
unzip mountain.zip -d scenedatabase/images
