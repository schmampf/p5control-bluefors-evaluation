from integration.files import Files

obj = Files(name="MyBase")

# print(obj.showData())

obj.saveData("test_data.pickle")
obj.loadData("test_data.pickle")
print(obj.showData())