function getSplit {
  local IMGDIR=$1
  local LBLDIR=$2
  local trainFile=$3
  local testFile=$4
  if [[ -e $trainFile ]] ; then
    echo "Error: $trainFile exists"
    exit 1
  fi
  if [[ -e $testFile ]] ; then
    echo "Error: $testFile exists"
    exit 2
  fi
  echo -e "# image, label" >> $trainFile
  echo -e "trainFiles=[" >> $trainFile
  echo -e "# image, label" >> $testFile
  echo -e "testFiles=[" >> $testFile
  for A in $IMGDIR/*
  do
    echo $A
    randn=$(( $RANDOM % 4 ))
    if (( $randn < 3 )) ; then
      outfile=$trainFile
    else 
      outfile=$testFile
    fi
    I=`basename $A`
    NAME="${I%.*}"
    echo -en "[\"$IMGDIR/${I}\", " >> ${outfile}
    echo -e  "\"$LBLDIR/${NAME}.png\",]," >> ${outfile}
  done
  echo -e "]" >> $trainFile
  echo -e "]" >> $testFile
}


echo "generating the split files"
#getSplit "$1" "$2" "$3" "$4"
getSplit "$1/images_hdr/" "$1/labels/" trainFiles.txt testFiles.txt
