#!/bin/bash
rm ./report/main.pdf

echo "Entering ./code"
cd code

echo "Deleting __pycache__"
rm -rf __pycache__

pyfiles=`find . | grep ".py"`
for file in $pyfiles
do
    echo -n "Processing $(basename $file)..."
    python $file
    echo " done"
done

echo "Processing is done, going back to root"
cd ..

echo "Entering ./report"
cd report

echo "Running pdflatex [1]..."
pdflatex -interaction=nonstopmode ./main.tex > /dev/null

echo "Running BibTex..."
bibtex main

echo "Running pdflatex [2]..."
pdflatex -interaction=nonstopmode ./main.tex > /dev/null

echo "Running pdflatex [3]..."
pdflatex -interaction=nonstopmode ./main.tex > /dev/null

echo "Copying report"
cp .main.pdf ../report.pdf

if [ $? -gt 0]
then
    echo "Something went wrong"
fi

echo "Going back to root"
cd ..