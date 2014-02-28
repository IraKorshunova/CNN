img_size = 1000
n_channels = 18
nkerns = [10, 20, 120]
recept_size = [31, 30]
pool_size = [5, 5]

#---------- dimensions
print 'c1:', nkerns[0], '@', img_size - recept_size[0] + 1, 'n_params:', nkerns[0] * (
recept_size[0] * n_channels + 1), 'prod', nkerns[0] * (img_size - recept_size[0] + 1)
print 's2:', nkerns[0], '@', (img_size - recept_size[0] + 1) / pool_size[0]

img_size = (img_size - recept_size[0] + 1) / pool_size[0]

print 'c3:', nkerns[1], '@', img_size - recept_size[1] + 1, 'nparams:', nkerns[1] * (
    recept_size[1] * nkerns[0] + 1), 'prod',  nkerns[1]*(img_size - recept_size[1] + 1)
print 's4:', nkerns[1], '@', (img_size - recept_size[1] + 1) / pool_size[1]

img_size = (img_size - recept_size[1] + 1) / pool_size[1]

print 'c5:', nkerns[2], '@', img_size, 'n_params:', nkerns[2] * img_size * nkerns[1], 'prod', nkerns[2]*img_size
