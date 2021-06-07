#!/bin/bash

pyfiles=`find . | grep ".py"`

for file in $pyfiles
do
    echo -n "Processing $(basename $file)..."
    #python $file
    echo " done"
done

echo "Processing is done"
echo -n "Running pdflatex [1]..."
pdflatex -interaction=nonstopmode ./report/main.tex > /dev/null

if [ $? -gt 0 ]
then
    echo "LaTeX compilation failed, exiting"
    exit 1
fi

echo -n "Running BibTex..."
bibtex ./report/main.tex

if [ $? -gt 0]
then
    echo "BibTeX Error"
    exit 1
fi

echo -n "Running pdflatex [2]..."
pdflatex -interaction=nonstopmode ./report/main.tex > /dev/null

if [ $? -gt 0 ]
then
    echo "LaTeX compilation failed, exiting"
    exit 1
fi


echo -n "Running pdflatex [3]..."
pdflatex -interaction=nonstopmode ./report/main.tex > /dev/null

if [ $? -gt 0 ]
then
    echo "LaTeX compilation failed, exiting"
    exit 1
fi

