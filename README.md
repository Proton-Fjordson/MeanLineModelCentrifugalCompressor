# Mean Line model for centrifugal compressors
This was made using the litterature as a side project after working hours.
As such, it may still contains somes mistakes but preliminary tests show good enough results in comparison with the litterature.

# Caution notice
The documentation is almost up to date. However, it's not perfect in the losses.py file. A few corrections were done without updating the documentation and the sources of the litterature used.
The culprits are probably:
* BladeloadingLoss
* SkinfrictionLoss
* ClearanceLoss
* RecirculationLoss
* LeakageLoss
* DiscfrictionLoss
* MixingLoss
* VanelessdiffuserLoss
* VaneddiffuserLoss
* VoluteLoss
From what I also remember, stresses computation may be buggy.
Last updated: 23/08/2024 (dd/mm/yyyy)

# Dependancies
* Documentation
 * sphinx
 * sphinx autoapi.extension
* computation
 * Coolprop
 * Refprop
 * numpy
 * scipy
 * collections.abc
 * matplotlib
 * time
 * sys
If refprop is not available, it's possible to make the code work by modifying the backend calls of Coolprop. 
However, there's a bit of elbow grease due to the calls to get the isentropic expansion coefficient values here and there.
This could be updated in a future version.

# Disclaimers
Commits were used throughout the development but squashed for confidentiality reasons. 
No internal data was used to develop and validate the code but use cases were linked to projects of my company.
I couldn't find the time to write the tests yet... Shame on me.