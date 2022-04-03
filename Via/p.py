
# Bubble Sort 
# def bubblesort(list):
# # Swap the elements to arrange in order
#    for iter_num in range(len(list)-1,0,-1):
#       for idx in range(iter_num):
#          if list[idx]>list[idx+1]:
#             list[idx],list[idx+1] = list[idx+1],list[idx]

# list = [19,2,31,45,6,11,121,27]

# bubblesort(list)
# print(list)




# Merge Sort
# def merge_sort(unsorted_list):
#    if len(unsorted_list) <= 1:
#       return unsorted_list
# # Find the middle point and devide it
#    middle = len(unsorted_list) // 2
#    left_list = unsorted_list[:middle]
#    right_list = unsorted_list[middle:]

#    left_list = merge_sort(left_list)
#    right_list = merge_sort(right_list)
#    return list(merge(left_list, right_list))

# # Merge the sorted halves
# def merge(left_half,right_half):
#    res = []
#    while len(left_half) != 0 and len(right_half) != 0:
#       if left_half[0] < right_half[0]:
#          res.append(left_half[0])
#          left_half.remove(left_half[0])
#       else:
#          res.append(right_half[0])
#          right_half.remove(right_half[0])
#    if len(left_half) == 0:
#       res = res + right_half
#    else:
#       res = res + left_half
#    return res
# unsorted_list = [64, 34, 25, 12, 22, 2 ,11, 90, 1]
# print(merge_sort(unsorted_list))



# Insertion Sort
# def insertion_sort(InputList):
#    for i in range(1, len(InputList)):
#       j = i-1
#       nxt_element = InputList[i]  
# # Compare the current element with next one
#    while (InputList[j] > nxt_element) and (j >= 0):
#       InputList[j+1] = InputList[j]
#       j=j-1
#    InputList[j+1] = nxt_element
# list = [19,2,31,45,30,11,121,27]
# insertion_sort(list)
# print(list)



class Node(data):
    def __init__(self,data):
        self.data = data
        self.next = None

class Linked():
    def __init__(self):
        self.data = None


    def add(self,data):
        node = Node(data)
        ß
        if self.next == None:

