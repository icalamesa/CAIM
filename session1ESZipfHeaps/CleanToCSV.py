with open('CountNewsOut') as infile, open('CountNewsOut.csv', 'w') as outfile:
  for line in infile:
    word = line.split(', ', 1)[1]
    print(word)
    if not any(i.isdigit() for i in word) and len(word)>1:
      outfile.write(line)