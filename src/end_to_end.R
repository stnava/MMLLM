#!/usr/bin/env Rscript
args<-commandArgs(TRUE)

whichk <- args[1]
istest = FALSE
# 1. denoise the image
# 2. segment it
# 3. reorient
# 4. landmark

############# setup
Sys.setenv("TF_NUM_INTEROP_THREADS"=4)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=4)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=4)
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


orinetInference <- function( template, target, mdlfn, doreo=NA, temspacescale, verbose=TRUE ) {

  # rotation template
  newspc = antsGetSpacing( template ) * temspacescale
  reoTemplateOrinet = resampleImage( template, newspc ) %>%
    iMath("PadImage", 32 )%>%
    iMath("Normalize")

  if ( is.na( doreo ) ) print( reoTemplateOrinet )
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
    if ( is.na( doreo ) ) {
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation' )
    } else if ( doreo %in% c(0,1,2) ) {
      reo = reorientImage( oimg , axis1 = doreo )
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation', initialTransform=reo$txfn )
    } else { # random initialization
      randtx = randomAffineImage( oimg, transformType = "Rigid", sdAffine = 5, interpolation = "linear" )[[2]]
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation', initialTransform=randtx )
    }
    initialTx = imglowreg$fwdtransforms
    orinetimg = imglowreg$warpedmovout %>% iMath("Normalize")
    orinetimg = smoothImage( orinetimg, 1.0, sigmaInPhysicalCoordinates = FALSE ) %>% iMath("Normalize")
    # END: a critical section of code that should match the training code

  imgarr = array( as.array( iMath(orinetimg, "Normalize") ), dim=c(1,dim(orinetimg),1) )
  with(tf$device("/cpu:0"), {
    predRot <- predict( orinet, tf$cast( imgarr, mytype), batch_size = 1 )
    })

  mm = matrix( predRot[1,], nrow=3, byrow=F)
  mmm = cbind( mm, pracma::cross( mm[,1], mm[,2] ) )
  mm = polarDecomposition( mmm )$Z
  locparams = getAntsrTransformParameters( myaff )
  locparams[1:9] = mm
  setAntsrTransformParameters( myaff, locparams )
  qreg = antsRegistration( reoTemplateOrinet, orinetimg, "Rigid", initialTransform=myaff )
  qreg = antsRegistration( reoTemplateOrinet, orinetimg, "Similarity", initialTransform=qreg$fwdtransforms )
  qreg = antsRegistration( reoTemplateOrinet, orinetimg, "Affine", initialTransform=qreg$fwdtransforms )
  img2LM = qreg$warpedmovout
  mymi = antsImageMutualInformation( iMath(reoTemplateOrinet,"Normalize"), iMath(img2LM,"Normalize") )
  antsImageWrite( iMath(reoTemplateOrinet,"Normalize"), '/tmp/temp0.nii.gz' )
  antsImageWrite( iMath(img2LM,"Normalize"), '/tmp/temp1.nii.gz' )
  mymsq = mean( abs( iMath(reoTemplateOrinet,"Normalize") - iMath(img2LM,"Normalize") ) )
  if ( verbose )
    print( paste( "MI:", mymi, "MSQ:", mymsq, " reo: ", doreo, 'mdl:', mdlfn ) )
  return( list( c(qreg$fwdtransforms,initialTx),  mymsq , mymi ) )
  }



orinetInferenceMulti <- function( template, target, orinetmdlfn, temspacescale, searchR=c(NA) ) {
  currentMI = Inf
  currentMSQ = Inf
  for ( rr in searchR ) {
    transform = orinetInference( reoTemplate, oimg, orinetmdlfn, doreo=rr, temspacescale=temspacescale, verbose=T )
    # if ( transform[[2]]  < currentMSQ ) {
    if ( transform[[3]]  < currentMI ) {
      transformX = transform[[1]]
      currentMSQ = transform[[2]]
      currentMI = transform[[3]]
    }
  }
  return( ( transformX ) )
}

lowerTrunc=1e-6
upperTrunc=0.98
outdir = "evaluation_results/"
if ( is.na( whichk ) ) {
  print("pass filename as argument")
  demog=read.csv('inferencemodels/mouse_test_data.csv')
  whichk = sample(1:nrow(demog),1)
  whichk = paste0( "bigdata/",demog$reo[whichk] )
  whichk = gsub( "_reorient.nii.gz", ".nii.gz", whichk )
#  q("no")
}
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_heat.nii.gz" )
if ( file.exists( outfn ) ) {
  print(paste("We believe this one is already done:",outfn))
  q("no")
}
print( paste("Begin:", whichk ) )
oimg = antsImageRead( whichk )
##########################################################################################################
# 1. denoise - p = 1 or p = 2 seem to be good - should optimize for all and implement via unet for speed
# img = denoiseImage( oimg, noiseModel='Gaussian', p = 2 ) # slower, maybe better
img = denoiseImage( oimg, noiseModel='Gaussian' ) # slower, maybe better

# 2. segment it
img = iMath( img, "TruncateIntensity", lowerTrunc, upperTrunc ) %>% iMath("Normalize")
skullmdlfn = "inferencemodels/mouse_skull_seg_from_ct_3.h5"
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
skullout = predict( unetSkull, tf$cast( iarr, mytype), batch_size = 1 )
segimg = as.antsImage( skullout[1,,,,1] ) %>% antsCopyImageInfo2( img )
segimg_bin = thresholdImage( segimg, 0.9, 1.0 ) %>% iMath("GetLargestComponent") # segimg is sufficient for LM


# 3. reorient to template space
# now do the same in the reo space
reoTemplate = antsImageRead( "templateImage.nii.gz" )
reoTemplatePad = iMath( reoTemplate, "PadImage",8)
reoSkull = antsImageRead( "templateImage_skull.nii.gz")
reoSkullPad = iMath( reoSkull, "PadImage",8)
onetmdl = "inferencemodels/mouse_rotation_3D_GPU_update0locsd5.h5"
# onetmdl = "/tmp/temp.h5"
templateTx = orinetInferenceMulti( reoTemplate, img, onetmdl, temspacescale=2.76, searchR=c(NA, 99) )

reoTemplate = antsImageRead( "templateImage.nii.gz" )
newspc = antsGetSpacing( reoTemplate ) * 1.22 # and pad by 8
reoTemplatePad = resampleImage( reoTemplate, newspc ) %>%
  iMath("PadImage", 8 )%>%
  iMath("Normalize")


iTx = antsApplyTransforms( reoTemplatePad, img, templateTx )
iSeg = antsApplyTransforms( reoTemplatePad, segimg, templateTx )

# iTx = antsImageRead( "preprocessed4/158_19__rec_reorient.nii.gz" )
# iSeg = antsImageRead( "preprocessed4/158_19__rec_reoskullprob.nii.gz" ) %>% thresholdImage(0.9,1) %>% iMath("GetLargestComponent")
print( reoTemplatePad )
mival = antsImageMutualInformation( reoTemplatePad, iTx )
print( paste("Registration:", mival ) )
# mival seems to be < -0.15 if its a reasonable result - usually around -0.2
# should visually check this result - hard to verify will work for *all* images
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_reorient.nii.gz" )
antsImageWrite( iTx, outfn )
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_skullprob.nii.gz" )
antsImageWrite( iSeg, outfn )
lmfn = gsub( "rec.nii.gz", "rec-LM.nii.gz", whichk )
lms = getCentroids( antsImageRead( lmfn ) )[,1:3]
lmsmap = antsApplyTransformsToPoints( 3, data.matrix(lms), rev(templateTx), whichtoinvert=c(TRUE,TRUE) )
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_pts_ground_truth.csv"  )
write.csv( lmsmap, outfn, row.names=FALSE )
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_reorient.png"  )
plot(  iTx, axis=3, nslices=21, ncolumns=7, alpha=0.5, outname=outfn )
print("plot done")
#   4.2 - architecture : deep lm network with mask ( uses heatmap )
#   4.3 - this network was trained with images in [0,255] intensity range (oops)
#   4.4 - input images should reorient to templateSpc * 1.22 padded ( step 2 )
##### start with architecture
gaussIt=FALSE
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
if ( gaussIt ) {
  findpoints = deepLandmarkRegressionWithHeatmaps( unetLM, activation='none', theta=NA, useMask=gaussIt )
  load_model_weights_hdf5( findpoints, 'inferencemodels/autopointsupdate_192_weights_3d_GPUsigmoidHRMask2.h5' )
} else {
  findpoints = deepLandmarkRegressionWithHeatmaps( unetLM, activation='none', theta=NA, useMask=gaussIt )
  load_model_weights_hdf5( findpoints, 'inferencemodels/autopointsupdate_192_weights_3d_GPUsigmoidHR0.h5' )
}
######### prepare the input data
# here is the heatmap
print( paste( "LM with mask?",  gaussIt ) )
tempmask = exp( -1.0 * iMath( iSeg, "D" ) /0.5 )
imgarr = array( as.array( iTx ), c( 1, dim(iTx), 1 ) )
segarr = array( as.array( tempmask ), c( 1, dim(iTx), 1 ) )
ccarr = array( dim=c( 1, dim(iTx), 3  ) )
mycc = coordinateImages( iTx )
for ( j in 1:3 ) ccarr[1,,,,j] = as.array( mycc[[j]] )
inflist = list(  tf$cast( imgarr, mytype), tf$cast( ccarr, mytype), tf$cast( segarr, mytype) )
if ( ! gaussIt ) inflist = inflist[1:2]
pointsoutte <- predict( findpoints, inflist, batch_size = 1 )
ppts = pointsoutte[[2]][1,,]
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_reorientpts.csv" )
write.csv( ppts, outfn, row.names=FALSE )
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_reorientpts.nii.gz" )
ptsi = makePointsImage( ppts, iTx * 0 + 1, 0.35 )
antsImageWrite( ptsi, outfn )
# get the sum of the locating maps
locmaps = tf$reduce_sum( pointsoutte[[1]] , axis=4L )
locmap = as.antsImage( as.array( locmaps )[1,,,] ) %>% antsCopyImageInfo2( reoTemplate )
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_heat.nii.gz" )
antsImageWrite( locmap, outfn )

# validation
denom  = norm( data.matrix( lmsmap ) )
numer = norm( data.matrix( lmsmap ) - ppts )
evaldfresult = data.frame( id= basename( whichk ) %>% tools::file_path_sans_ext( T ), errnorm =numer/denom  )
print( evaldfresult  )
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_eval.csv" )
write.csv( evaldfresult, outfn, row.names=FALSE )
