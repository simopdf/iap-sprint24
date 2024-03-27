from PIL import Image


for i in range(10):
    im = Image.open("sphere_celeste_test {}.ppm".format(i))
    im.save("test {:02d}.jpg".format(i))

    print("{:.1%} completed".format(i//100))



###  A faire pour eviter les problèmes d'ordre des images
"""To add leading zeros to numbers in Python, you can use the format() function, f-string technique, rjust() function, or zfill() function. 
Each method allows you to specify the total number of digits in the output, automatically adding the necessary zeros to the 
left of the number to meet the desired width.   "{:06d}".format(3) -> 000003"""