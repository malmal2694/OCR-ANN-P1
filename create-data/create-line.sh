#! /bin/sh

line_length=15
# Create lines; each line of the new file represent a combination of words in complete-shuffled.wordlist file
awk -v ll="$line_length" '{for (i=1; i<=NF; ++i)printf "%s%s", $i, i % 15 ? " ": "\n"}i % 15{print ""}' complete-shuffled.wordlist > data/lines.txt
split -n 4 -d data/lines.txt data/sublines.txt

