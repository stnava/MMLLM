#!/usr/bin/env Rscript
args<-commandArgs(TRUE)

whichk <- args[1]

# 1. denoise the image
# 2. segment it
# 3. reorient
# 4. landmark
istest=FALSE
############# setup
Sys.setenv("TF_NUM_INTEROP_THREADS"=24)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=24)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=24)
mytype = "float32"
library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )
library( reticulate )
library( ggplot2 )
library( tfdatasets )
library( glue )


orinetInference <- function( template, target, mdlfn, doreo=NULL, newway=FALSE, verbose=TRUE ) {

  # rotation template
  newspc = antsGetSpacing( template ) * 2.0
  reoTemplateOrinet = resampleImage( template, newspc ) %>%
    iMath("PadImage", 32 )%>%
    iMath("Normalize")

  orinet = createResNetModel3D(
      list(NULL,NULL,NULL,1),
      inputScalarsSize = 0,
      numberOfClassificationLabels = 6,
      layers = 1:4,
      residualBlockSchedule = c(3, 4, 6, 3),
      lowestResolution = 64,
      cardinality = 1,
      squeezeAndExcite = TRUE,
      mode = "regression"
    )
  load_model_weights_hdf5( orinet, mdlfn )

  myaff = randomAffineImage( reoTemplateOrinet, "Rigid", sdAffine = 0 )[[2]]
  idparams = getAntsrTransformParameters( myaff )
  fixparams = getAntsrTransformFixedParameters( myaff )
  templateCoM = getCenterOfMass( reoTemplateOrinet )


  # BEGIN: a critical section of code that should match the training code
  oimg = iMath(target, "Normalize")
    if ( is.null( doreo ) ) {
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation' )
    } else if ( doreo %in% c(0,1,2) ) {
      reo = reorientImage( oimg , axis1 = doreo )
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation', initialTransform=reo$txfn )
    } else { # random initialization
      randtx = randomAffineImage( oimg, transformType = "Rigid", sdAffine = 5, interpolation = "linear" )[[2]]
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation', initialTransform=randtx )
    }
    initialTx = imglowreg$fwdtransforms
    imglow = imglowreg$warpedmovout
    orinetimg = smoothImage( imglow, 1.5, sigmaInPhysicalCoordinates = FALSE ) %>% iMath("Normalize")

  # END: a critical section of code that should match the training code

  imgCoM = getCenterOfMass( iMath(orinetimg, "Normalize") )
  imgarr = array( as.array( iMath(orinetimg, "Normalize") ), dim=c(1,dim(orinetimg),1) )
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
  # myaff = composeAntsrTransforms( list( myaff, initialTx ) )
  # rotated = applyAntsrTransformToImage( myaff, orinetimg, reoTemplate )
  qreg = antsRegistration( reoTemplateOrinet, imglow, "Rigid", initialTransform=myaff )
  qreg = antsRegistration( reoTemplateOrinet, imglow, "Similarity", initialTransform=qreg$fwdtransforms )
  qreg = antsRegistration( reoTemplateOrinet, imglow, "Affine", initialTransform=qreg$fwdtransforms )
  img2LM = qreg$warpedmovout
  mymi = antsImageMutualInformation( iMath(reoTemplateOrinet,"Normalize"), iMath(img2LM,"Normalize") )
  mymsq = mean( abs( iMath(reoTemplateOrinet,"Normalize") - iMath(img2LM,"Normalize") ) )
  if ( verbose )
    print( paste( "MI:", mymi, "MSQ:", mymsq, " reo: ", doreo, 'mdl:', mdlfn ) )
  return( list( c( qreg$fwdtransforms, initialTx ),  mymsq , mymi ) )
  }



orinetInferenceMulti <- function( template, target, orinetmdlfn ) {
  currentMI = Inf
  currentMSQ = Inf
  for ( rr in c(NULL,0,1,2,3:12) ) {
    transform = orinetInference( reoTemplate, oimg, orinetmdlfn, doreo=rr, newway=FALSE, verbose=T )
    if ( transform[[2]]  < currentMSQ  &   transform[[3]]  < currentMI ) {
      transformX = transform[[1]]
      currentMSQ = transform[[2]]
      currentMI = transform[[3]]
    }
  }
  return( transformX )
}

############# none of these used in train/test
valids = c(
  "152-27__rec",
  "152-29__rec",
  "152-30__rec",
  "skull_152_8__rec",
  "skull_155_10__rec",
  "skull_155_11__rec",
  "skull_155_13__rec",
  "skull_155_14__rec",
  "skull_155_17__rec",
  "skull_155_18__rec",
  "skull_155_19__rec",
  "skull_160_1__rec",
  "skull_160_6__rec" )


lowerTrunc=1e-6
upperTrunc=0.98
fn = Sys.glob( paste0( "bigdata/*rec.nii.gz" ) )
if ( is.na( whichk ) ) fn = sample( fn, 1 )
fn = Sys.glob( paste0( "bigdata/155-35**rec.nii.gz" ) )
fn = Sys.glob( paste0( "bigdata/skull_155_21_*rec.nii.gz" ) ) # 159-25__rec_
outfn = paste0( "preprocessed2/", basename( fn ) %>% tools::file_path_sans_ext( T ), "_reorient.png"  )
if ( file.exists( outfn ) ) q("no")
print( paste("Begin:", fn ) )
# istest = TRUE
# if ( istest ) fn = 'bigdata/160-41__rec.nii.gz'
oimg = antsImageRead( fn )

# 1. denoise - p = 1 or p = 2 seem to be good - should optimize for all and implement via unet for speed
# img = denoiseImage( oimg, noiseModel='Gaussian', p = 2 ) # slower, maybe better
img = denoiseImage( oimg, noiseModel='Gaussian' ) # slower, maybe better

# 2. segment it
img = iMath( img, "TruncateIntensity", lowerTrunc, upperTrunc ) %>% iMath("Normalize")
skullmdlfn = "models/mouse_skull_seg_from_ct_3.h5"
unetSkull = createUnetModel3D(
       list( NULL, NULL, NULL, 1),
       numberOfOutputs = 1,
       numberOfLayers = 4,
       numberOfFiltersAtBaseLayer = 32,
       convolutionKernelSize = 3,
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0.0,
       weightDecay = 0,
       additionalOptions = "nnUnetActivationStyle",
       mode = c("sigmoid") )
load_model_weights_hdf5( unetSkull, skullmdlfn )
iarr = array( as.array( img ), dim = c( 1, dim( img ), 1 ) )
skullout <- predict( unetSkull, tf$cast( iarr, mytype), batch_size = 1 )
segimg = as.antsImage( skullout[1,,,,1] ) %>% antsCopyImageInfo2( img )
segimg_bin = thresholdImage( segimg, 0.6, 1.0 ) %>% iMath("GetLargestComponent") # segimg is sufficient for LM

# 3. reorient to template space
# now do the same in the reo space
reoTemplate = antsImageRead( "templateImage.nii.gz" ) %>% iMath("PadImage",8)
reoSkull = antsImageRead( "templateImage_skull.nii.gz") %>% iMath("PadImage",8)
templateTx = orinetInferenceMulti( reoSkull, segimg_bin, "models/mouse_rotation_3D_GPU_update1locsd5.h5" )
iTx = antsApplyTransforms( reoTemplate, img, templateTx )
iTx2 = antsApplyTransforms( reoTemplate, oimg, templateTx )
iSeg = antsApplyTransforms( reoTemplate, segimg, templateTx )
mival = antsImageMutualInformation( reoTemplate, iTx )
print( paste("Registration:", mival ) )
# mival seems to be < -0.15 if its a reasonable result - usually around -0.2
# should visually check this result - hard to verify will work for *all* images
outfn = paste0( "preprocessed2/", basename( fn ) %>% tools::file_path_sans_ext( T ), "_reorient.nii.gz" )
antsImageWrite( iTx, outfn )
outfn = paste0( "preprocessed2/", basename( fn ) %>% tools::file_path_sans_ext( T ), "_skullprob.nii.gz" )
antsImageWrite( iSeg, outfn )
lmfn = gsub( "rec.nii.gz", "rec-LM.nii.gz", fn )
lms = getCentroids( antsImageRead( lmfn ) )[,1:3]
lmsmap = antsApplyTransformsToPoints( 3, data.matrix(lms), rev(templateTx), whichtoinvert=c(TRUE,TRUE) )
outfn = paste0( "preprocessed2/", basename( fn ) %>% tools::file_path_sans_ext( T ), "_pts.csv"  )
write.csv( lmsmap, outfn, row.names=FALSE )
outfn = paste0( "preprocessed2/", basename( fn ) %>% tools::file_path_sans_ext( T ), "_reorient.png"  )
plot(  iTx, axis=3, nslices=21, ncolumns=7, alpha=0.5 )

# 4. the landmarking problem - important to have preprocessing right
#   4.1 - need a heatmap of the skull ( comes from segmentation above )
#   4.2 - architecture : deep lm network with mask ( uses heatmap )
#   4.3 - this network was trained with images in [0,255] intensity range (oops)
#   4.4 - input images should reorient to templateSpc * 1.22 padded ( step 2 )
##### start with architecture
gaussIt=TRUE
unetLM = createUnetModel3D(
       list( NULL, NULL, NULL, 1),
       numberOfOutputs = 55,
       numberOfLayers = 5,
       numberOfFiltersAtBaseLayer = 32,
       convolutionKernelSize = 3,
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0.0,
       weightDecay = 0,
       additionalOptions = "nnUnetActivationStyle",
       mode = c("sigmoid")
     )
findpoints = deepLandmarkRegressionWithHeatmaps( unetLM, activation='none', theta=NA, useMask=gaussIt )
load_model_weights_hdf5( findpoints, 'models/autopointsupdate_192_weights_3d_GPUsigmoidHRMask2.h5' )
######### prepare the input data
iTx2T = iMath( iTx2, 'TruncateIntensity', 1e-6, 0.98 )

if ( istest ) {
  skullfn = 'preprocessed/160-41__rec-pro-skull.nii.gz'
  iSeg = antsImageRead( skullfn )
  iTx2T = antsImageRead( "preprocessed/160-41__rec-pro-reorient.nii.gz")
}
# here is the heatmap
tempmask = thresholdImage(iSeg,0.85,1.0) %>% iMath("GetLargestComponent")
tempmask = exp( -1.0 * iMath( tempmask, "D" ) / 0.5 )

imgarr = array( as.array( iTx2T ), c( 1, dim(iTx2T), 1 ) )
segarr = array( as.array( tempmask ), c( 1, dim(iTx2T), 1 ) )
ccarr = array( dim=c( 1, dim(iTx), 3  ) )
mycc = coordinateImages( iTx )
for ( j in 1:3 ) ccarr[1,,,,j] = as.array( mycc[[j]] )
inflist = list(  tf$cast( imgarr, mytype), tf$cast( ccarr, mytype), tf$cast( segarr, mytype) )
pointsoutte <- predict( findpoints, inflist, batch_size = 1 )
ppts = pointsoutte[[2]][1,,]
ptsi = makePointsImage( ppts, iTx * 0 + 1, 0.5 )
antsImageWrite( iTx, '/tmp/temp.nii.gz' )
antsImageWrite( ptsi, '/tmp/tempp.nii.gz' )
# get the sum of the locating maps
locmaps = tf$reduce_sum( pointsoutte[[1]] , axis=4L )
locmap = as.antsImage( as.array( locmaps )[1,,,] ) %>% antsCopyImageInfo2( reoTemplate )
antsImageWrite( locmap, '/tmp/temph.nii.gz' )



# validation
denom  = norm( data.matrix( lmsmap ) )
numer = norm( data.matrix( lmsmap ) - ppts )
print( numer/denom  )
###################################################
ptsi = makePointsImage( lmsmap, iTx * 0 + 1, 0.5 )
antsImageWrite( ptsi, '/tmp/temppgt.nii.gz' )
