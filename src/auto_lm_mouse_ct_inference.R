Sys.setenv("TF_NUM_INTEROP_THREADS"=12)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=12)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=12)

read.fcsv<-function( x, skip=3 ) {
  df = read.table( x, skip=skip, sep=',' )
  colnames( df ) = c("id","x","y","z","ow","ox","oy","oz","vis","sel","lock","label","desc","associatedNodeID")
  return( df )
  }

library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )
library( reticulate )
library( ggplot2 )
library( tfdatasets )
library( glue )
mytype = "float32"

reoTemplate = antsImageRead( "templateImage.nii.gz" ) # antsImageClone( imgListTest[[templatenum]] )
ptTemplate = data.matrix( read.csv( "templatePoints.csv" ) )# ptListTest[[templatenum]]
locdim = dim(reoTemplate)

fnsNew = Sys.glob("data_landmark_new_04_13_2021/*_.nii.gz")


# this is inference code
orinet =  createResNetModel3D(
       list(NULL,NULL,NULL,1),
       inputScalarsSize = 0,
       numberOfClassificationLabels = 6,
       layers = 1:4,
       residualBlockSchedule = c(3, 4, 6, 3),
       lowestResolution = 16,
       cardinality = 1,
       squeezeAndExcite = TRUE,
       mode = "regression")
load_model_weights_hdf5( orinet, "models/mouse_rotation_3D_GPU2.h5" )

unet = createUnetModel3D(
       list( NULL, NULL, NULL, 1),
       numberOfOutputs = 55,
       numberOfLayers = 4,
       numberOfFiltersAtBaseLayer = 32,
       convolutionKernelSize = 3,
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0,
       weightDecay = 0,
       additionalOptions = "nnUnetActivationStyle",
       mode = c("regression")
     )
unet = keras_model( unet$inputs, tf$nn$sigmoid( unet$outputs[[1]] ) )
findpoints = deepLandmarkRegressionWithHeatmaps( unet, activation='relu', theta=NA )
load_model_weights_hdf5( findpoints,   "models/autopointsfocused_sigmoid_128_weights_3d_checkpoints5_GPU2.h5" )
myaff = randomAffineImage( reoTemplate, "Rigid", sdAffine = 0 )[[2]]
idparams = getAntsrTransformParameters( myaff )
fixparams = getAntsrTransformFixedParameters( myaff )
templateCoM = getCenterOfMass( reoTemplate )

whichk = sample( 1:length(fnsNew), 1 )
whichk = 6
print( whichk )
oimg = antsImageRead( fnsNew[whichk] ) %>% resampleImage( locdim, useVoxels=TRUE)
img = antsImageClone( oimg )
invisible( antsCopyImageInfo2( img, reoTemplate ) )
imgCoM = getCenterOfMass( iMath(img, "Normalize") )
imgarr = array( as.array( iMath(img, "Normalize") ), dim=c(1,locdim,1) )
print("deep rigid")
with(tf$device("/cpu:0"), {
      predRot <- predict( orinet, tf$cast( imgarr, mytype), batch_size = 1 )
    })
mm = matrix( predRot[1,], nrow=3, byrow=F)
mmm = cbind( mm, pracma::cross( mm[,1], mm[,2] ) )
mm = polarDecomposition( mmm )$Z
locparams = getAntsrTransformParameters( myaff )
locparams[1:9] = mm
locparams[10:length(locparams)] = (imgCoM - templateCoM )
setAntsrTransformParameters( myaff, locparams )
setAntsrTransformFixedParameters( myaff, templateCoM )
rotated = applyAntsrTransformToImage( myaff, img, reoTemplate )
print("classical rigid")
qreg = antsRegistration( reoTemplate, img, "Rigid", initialTransform=myaff )
print("classical sim")
qreg = antsRegistration( reoTemplate, img, "Similarity", initialTransform=qreg$fwdtransforms )
print("classical aff")
qreg = antsRegistration( reoTemplate, img, "Affine", initialTransform=qreg$fwdtransforms )
img2LM = qreg$warpedmovout
print( antsImageMutualInformation( reoTemplate, img2LM) )

img2LMcoords = coordinateImages( img2LM * 0 + 1 )
mycc = array( dim = c( 1, dim( img2LM ), 3 ) )
for ( jj in 1:3 ) mycc[1,,,,jj] = as.array( img2LMcoords[[jj]] )
imgarr[1,,,,1] = as.array( iMath( img2LM, "Normalize" ) )
telist = list(  tf$cast( imgarr, mytype), tf$cast( mycc, mytype) )
with(tf$device("/cpu:0"), {
      pointsoutte <- predict( findpoints, telist, batch_size = 1 )
      })
ptp = as.array(pointsoutte[[2]])[1,,]
ptimg = makePointsImage( ptp, img2LM*0+1, radius=1 )  %>% iMath("GD",3)
print(sort(unique(ptimg)))
# plot( img2LM, ptimg, nslices = 21, ncolumns = 7 )
ptmask = thresholdImage( ptimg, 1, 2 )
ptmaskdil = iMath( ptmask, "MD", 8 )
#plot(
#  cropImage(img2LM,ptmaskdil),
#  cropImage(ptimg*thresholdImage(ptimg,1,2),ptmaskdil), nslices = 21, ncolumns = 7 )
# now we transform points to original (rotational) space
ptpb = antsApplyTransformsToPoints( 3, ptp, qreg$fwdtransforms, whichtoinvert=c(FALSE) )
ptimg2 = makePointsImage( ptpb, img*0+1, radius=1 ) %>% iMath("GD",4)
# plot( img, ptimg, nslices = 21, ncolumns = 7, axis=1 )
# but the original image is in yet another space.
# there is a purely mathematical conversion that could be applied just based
# on the header space differences but we dont have that code on hand - this is
# why we convert to image space then back to point space.   note, however,
# that - for morphometry - rotations do not matter.  and global scale is just
# based on the spacing.
invisible( antsCopyImageInfo2( ptimg2, oimg ) )
ptsOriginalSpace = getCentroids( ptimg2 )[,1:img@dimension]
trad = sqrt( sum( antsGetSpacing( oimg )^2 ) )
ptsi = makePointsImage( ptsOriginalSpace, oimg*0+1, radius = trad ) %>% iMath( "GD", 1 )
# plot( oimg, ptsi, nslices = 21, ncolumns = 7, axis=1 )
write.csv( ptsOriginalSpace, 'temp.csv', row.names=F )
# NOTE: it may be better to apply this mapping to the coordinate space heat
# maps output from the unet, backtransform them to the original space, and
# then estimate the points.
# the first output from the unet is the heat-map images.
# pointsoutte[[1]]
#
# compare to tru points
trupts = read.fcsv( paste0( fnsNew[whichk], ".fcsv" ) )
trupts = data.matrix( trupts[,c("x","y","z")] )
distancesByPoint = rep( NA, nrow( trupts ) )
for ( k in 1:nrow( trupts ) ) {
  distancesByPoint[k] = sqrt( sum( ( trupts[k,] - ptsOriginalSpace[k,] )^2 ) )
}
print( distancesByPoint )

print( mean( distancesByPoint ) )

worstinds = head( rev(order(distancesByPoint)), 6 )
print( " bad ones " )
print( worstinds )
print( distancesByPoint[ worstinds ] )

layout( matrix(1:2,nrow=2))
# these are the tru points
ptsitru = makePointsImage( trupts, oimg*0+1, radius = trad ) %>% iMath( "GD", 1 )
binmask = maskImage( ptsitru, ptsitru, level = worstinds, binarize= TRUE  )
binmaskdil = iMath(binmask, "MD", 6 )
ptsitrumsk = maskImage( ptsitru, ptsitru, level = worstinds, binarize=FALSE )
plot(
  cropImage(oimg,binmaskdil),
  cropImage(ptsitrumsk,binmaskdil), nslices = 28, ncolumns = 14, axis=1 )

#
# these are the est points
ptsiest = makePointsImage( ptsOriginalSpace, oimg*0+1, radius = trad ) %>% iMath( "GD", 1 )
ptsiestmsk = maskImage( ptsiest, ptsiest, level = worstinds, binarize=FALSE )
plot(
  cropImage(oimg,binmaskdil),
  cropImage(ptsiestmsk,binmaskdil), nslices = 28, ncolumns = 14, axis=1 )


if ( FALSE ) {
  gg = generateData( batch_size = 1, mySdAff=0, whichSample=219)
  with(tf$device("/cpu:0"), {
        pointsoutte <- predict( findpoints, gg[1:2], batch_size = 1 )
        })
  ptp = as.array(pointsoutte[[2]])[1,,]
  img2LM = as.antsImage( gg[[1]][1,,,,1] ) %>% antsCopyImageInfo2( reoTemplate )
  ptimg = makePointsImage( ptp, img2LM*0+1, radius=1 )  %>% iMath("GD",3)
  sort(unique(ptimg))
  plot( img2LM, ptimg, nslices = 21, ncolumns = 7 )
}


# 51 52 29  1  2 53
# 51 29 52  1  2 53
# 51 29 52  1 53  2
