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
  padval = 32
  newspc = antsGetSpacing( template ) * temspacescale
  reoTemplateOrinet = resampleImage( template, newspc ) %>%
    iMath("PadImage", padval )%>%
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
    matparams = head( getAntsrTransformParameters( readAntsrTransform( initialTx ) ), 9 )
    mymat = matrix( matparams, nrow=3 )
    mytrans = tail( getAntsrTransformParameters( readAntsrTransform( initialTx ) ), 3 )
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
  locparams[1:9] = ( mm %*% ( mymat ) ) # compose the rotational components
  locparams[10:12] = mytrans # use the initial transform's translation
  setAntsrTransformParameters( myaff, locparams )
  qreg = antsRegistration( reoTemplateOrinet, target, "Rigid", initialTransform=myaff )
  qreg = antsRegistration( reoTemplateOrinet, target, "Similarity", initialTransform=qreg$fwdtransforms )
  qreg = antsRegistration( reoTemplateOrinet, target, "Affine", initialTransform=qreg$fwdtransforms )
  img2LM = qreg$warpedmovout
  mymi = antsImageMutualInformation( iMath(reoTemplateOrinet,"Normalize"), iMath(img2LM,"Normalize") )
  mymsq = mean( abs( iMath(reoTemplateOrinet,"Normalize") - iMath(img2LM,"Normalize") ) )
  if ( verbose )
    print( paste( "MI:", mymi, "MSQ:", mymsq, " reo: ", doreo, 'mdl:', mdlfn ) )
  return( list( qreg$fwdtransforms,  mymsq , mymi ) )
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
  return( transformX )
}



# assumes target image is reoriented to template space
mouseSkullCTLM <- function( targetImage,  modelFN ) {
  ##### start with architecture
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

  findpoints = deepLandmarkRegressionWithHeatmaps( unetLM, activation='none', theta=NA, useMask=FALSE )
  load_model_weights_hdf5( findpoints, modelFN )

  ######### prepare the input data
  # here is the heatmap
  imgarr = array( as.array( targetImage ), c( 1, dim(targetImage), 1 ) )
  ccarr = array( dim=c( 1, dim(targetImage), 3  ) )
  mycc = coordinateImages( targetImage )
  for ( j in 1:3 ) ccarr[1,,,,j] = as.array( mycc[[j]] )
  inflist = list(  tf$cast( imgarr, mytype), tf$cast( ccarr, mytype) )
  pointsoutte <- predict( findpoints, inflist, batch_size = 1 )
  ppts = pointsoutte[[2]][1,,]
  return( ppts )
  }

lowerTrunc=1e-6
upperTrunc=0.98
outdir = "evaluation_results2/"
if ( is.na( whichk ) ) {
  print("pass filename as argument")
  demog=read.csv('inferencemodels/mouse_test_data.csv')
  whichk = sample(1:nrow(demog),1)
  whichk = paste0( "bigdata/",demog$reo[whichk] )
  whichk = gsub( "_reorient.nii.gz", ".nii.gz", whichk )
#  q("no")
}
whichk = "bigdata/skull_155_19__rec.nii.gz"
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
reoTemplate = antsImageRead( "templateImage.nii.gz" ) %>%
  iMath("Normalize")
onetmdl = "inferencemodels/mouse_rotation_3D_GPU_update0locsd5.h5"
# onetmdl = "/tmp/temp.h5"
templateTx = orinetInferenceMulti( reoTemplate, img, onetmdl,
  temspacescale=2.76, searchR=c(NA,5:8) ) # this searches a few starting points
iTx = antsApplyTransforms( reoTemplate, img, templateTx )
iSeg = antsApplyTransforms( reoTemplate, segimg, templateTx )

mival = antsImageMutualInformation( reoTemplate, iTx )
print( paste("Registration:", mival ) )
# mival seems to be < -0.15 if its a reasonable result - usually around -0.2
# should visually check this result - hard to verify will work for *all* images
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_reorient.nii.gz" )
antsImageWrite( iTx, outfn )
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_skullprob.nii.gz" )
antsImageWrite( iSeg, outfn )

outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_reorient.png"  )
plot(  iTx, axis=3, nslices=21, ncolumns=7, alpha=0.5, outname=outfn )

ptmdl = 'inferencemodels/autopointsupdate_192_weights_3d_GPUsigmoidHR1.h5'
lmInTemplateSpace = mouseSkullCTLM( iTx,  ptmdl )
ptsi = makePointsImage( lmInTemplateSpace, iTx * 0 + 1, 0.35 )
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_reorientpts.nii.gz" )
antsImageWrite( ptsi, outfn )

# map back to original space
lmsmap = antsApplyTransformsToPoints( 3, data.matrix(lmInTemplateSpace), templateTx[1]  )
lmsmap = data.matrix(lmsmap)
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_pts.csv" )
write.csv( lmsmap, outfn, row.names=FALSE )
outfn = paste0( outdir, basename( whichk ) %>% tools::file_path_sans_ext( T ), "_pts.nii.gz" )
ptsi = makePointsImage( lmsmap, oimg * 0 + 1, 0.35 )
antsImageWrite( ptsi, outfn )
mycmd = paste( "snap -g ",  whichk, " -s ", outfn )
print( mycmd )
