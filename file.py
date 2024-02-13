def get_even(lst):
    ''' Функция создает список из четных чисел'''
    even_lst = []
    for elem in lst:
        if not elem % 2:
            even_lst.append(elem)
    return even_lst


# help(get_even)
print(get_even.__doc__)