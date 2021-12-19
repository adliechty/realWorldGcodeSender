import subprocess
import csv

results = []
for y in range (500, 3000, 100):
  print()
  xResults = []
  for x in range (1000, 2000, 5):
    result = subprocess.run(['python', 'realWorldGcodeSender.py', str(x), str(y)], stdout=subprocess.PIPE)
    if "Not found" in str(result.stdout):
      print(str(x) + "," + str(y) + ":not found")
      xResults.append(0)
    else:
      print(str(x) + "," + str(y) + ":found")
      xResults.append(1)
  results.append(xResults)

for result in results:
  print(result)
with open("new_file.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(results)
