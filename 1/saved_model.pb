??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??
?
dense_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_namedense_layer1/kernel
{
'dense_layer1/kernel/Read/ReadVariableOpReadVariableOpdense_layer1/kernel*
_output_shapes

:
*
dtype0
z
dense_layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namedense_layer1/bias
s
%dense_layer1/bias/Read/ReadVariableOpReadVariableOpdense_layer1/bias*
_output_shapes
:
*
dtype0
?
dense_layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_namedense_layer2/kernel
{
'dense_layer2/kernel/Read/ReadVariableOpReadVariableOpdense_layer2/kernel*
_output_shapes

:
*
dtype0
z
dense_layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namedense_layer2/bias
s
%dense_layer2/bias/Read/ReadVariableOpReadVariableOpdense_layer2/bias*
_output_shapes
:*
dtype0
?
dense_layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_namedense_layer3/kernel
{
'dense_layer3/kernel/Read/ReadVariableOpReadVariableOpdense_layer3/kernel*
_output_shapes

:
*
dtype0
z
dense_layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namedense_layer3/bias
s
%dense_layer3/bias/Read/ReadVariableOpReadVariableOpdense_layer3/bias*
_output_shapes
:
*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/dense_layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*+
shared_nameAdam/dense_layer1/kernel/m
?
.Adam/dense_layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer1/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dense_layer1/bias/m
?
,Adam/dense_layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer1/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*+
shared_nameAdam/dense_layer2/kernel/m
?
.Adam/dense_layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer2/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_layer2/bias/m
?
,Adam/dense_layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*+
shared_nameAdam/dense_layer3/kernel/m
?
.Adam/dense_layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer3/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dense_layer3/bias/m
?
,Adam/dense_layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer3/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*+
shared_nameAdam/dense_layer1/kernel/v
?
.Adam/dense_layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer1/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dense_layer1/bias/v
?
,Adam/dense_layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer1/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*+
shared_nameAdam/dense_layer2/kernel/v
?
.Adam/dense_layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer2/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/dense_layer2/bias/v
?
,Adam/dense_layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*+
shared_nameAdam/dense_layer3/kernel/v
?
.Adam/dense_layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer3/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dense_layer3/bias/v
?
,Adam/dense_layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer3/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:
*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?0
value?0B?0 B?0
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
?
$iter

%beta_1

&beta_2
	'decay
(learning_ratemSmTmUmVmWmXmYmZv[v\v]v^v_v`vavb
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
)non_trainable_variables
*layer_metrics
+metrics
trainable_variables

,layers
	variables
-layer_regularization_losses
	regularization_losses
 
_]
VARIABLE_VALUEdense_layer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdense_layer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
.non_trainable_variables
/layer_metrics
0metrics
trainable_variables

1layers
	variables
2layer_regularization_losses
regularization_losses
_]
VARIABLE_VALUEdense_layer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdense_layer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
3non_trainable_variables
4layer_metrics
5metrics
trainable_variables

6layers
	variables
7layer_regularization_losses
regularization_losses
_]
VARIABLE_VALUEdense_layer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdense_layer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
8non_trainable_variables
9layer_metrics
:metrics
trainable_variables

;layers
	variables
<layer_regularization_losses
regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
=non_trainable_variables
>layer_metrics
?metrics
 trainable_variables

@layers
!	variables
Alayer_regularization_losses
"regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

B0
C1
D2
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Etotal
	Fcount
G	variables
H	keras_api
D
	Itotal
	Jcount
K
_fn_kwargs
L	variables
M	keras_api
D
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

G	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

L	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1

Q	variables
??
VARIABLE_VALUEAdam/dense_layer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_layer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_layer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_layer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_layer3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_layer3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_layer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_layer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_layer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_layer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_layer3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_layer3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputdense_layer1/kerneldense_layer1/biasdense_layer2/kerneldense_layer2/biasdense_layer3/kerneldense_layer3/biasdense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_3848
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'dense_layer1/kernel/Read/ReadVariableOp%dense_layer1/bias/Read/ReadVariableOp'dense_layer2/kernel/Read/ReadVariableOp%dense_layer2/bias/Read/ReadVariableOp'dense_layer3/kernel/Read/ReadVariableOp%dense_layer3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp.Adam/dense_layer1/kernel/m/Read/ReadVariableOp,Adam/dense_layer1/bias/m/Read/ReadVariableOp.Adam/dense_layer2/kernel/m/Read/ReadVariableOp,Adam/dense_layer2/bias/m/Read/ReadVariableOp.Adam/dense_layer3/kernel/m/Read/ReadVariableOp,Adam/dense_layer3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp.Adam/dense_layer1/kernel/v/Read/ReadVariableOp,Adam/dense_layer1/bias/v/Read/ReadVariableOp.Adam/dense_layer2/kernel/v/Read/ReadVariableOp,Adam/dense_layer2/bias/v/Read/ReadVariableOp.Adam/dense_layer3/kernel/v/Read/ReadVariableOp,Adam/dense_layer3/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_4267
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_layer1/kerneldense_layer1/biasdense_layer2/kerneldense_layer2/biasdense_layer3/kerneldense_layer3/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/dense_layer1/kernel/mAdam/dense_layer1/bias/mAdam/dense_layer2/kernel/mAdam/dense_layer2/bias/mAdam/dense_layer3/kernel/mAdam/dense_layer3/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_layer1/kernel/vAdam/dense_layer1/bias/vAdam/dense_layer2/kernel/vAdam/dense_layer2/bias/vAdam/dense_layer3/kernel/vAdam/dense_layer3/bias/vAdam/dense/kernel/vAdam/dense/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_4382??
?
?
F__inference_dense_layer3_layer_call_and_return_conditional_losses_3568

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
5dense_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer3/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer3/kernel/Regularizer/SquareSquare=dense_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer3/kernel/Regularizer/Square?
%dense_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer3/kernel/Regularizer/Const?
#dense_layer3/kernel/Regularizer/SumSum*dense_layer3/kernel/Regularizer/Square:y:0.dense_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/Sum?
%dense_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer3/kernel/Regularizer/mul/x?
#dense_layer3/kernel/Regularizer/mulMul.dense_layer3/kernel/Regularizer/mul/x:output:0,dense_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/mulf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
F__inference_functional_1_layer_call_and_return_conditional_losses_3717

inputs
dense_layer1_3678
dense_layer1_3680
dense_layer2_3683
dense_layer2_3685
dense_layer3_3688
dense_layer3_3690

dense_3693

dense_3695
identity??dense/StatefulPartitionedCall?$dense_layer1/StatefulPartitionedCall?$dense_layer2/StatefulPartitionedCall?$dense_layer3/StatefulPartitionedCall?
$dense_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_layer1_3678dense_layer1_3680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer1_layer_call_and_return_conditional_losses_35022&
$dense_layer1/StatefulPartitionedCall?
$dense_layer2/StatefulPartitionedCallStatefulPartitionedCall-dense_layer1/StatefulPartitionedCall:output:0dense_layer2_3683dense_layer2_3685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer2_layer_call_and_return_conditional_losses_35352&
$dense_layer2/StatefulPartitionedCall?
$dense_layer3/StatefulPartitionedCallStatefulPartitionedCall-dense_layer2/StatefulPartitionedCall:output:0dense_layer3_3688dense_layer3_3690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer3_layer_call_and_return_conditional_losses_35682&
$dense_layer3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-dense_layer3/StatefulPartitionedCall:output:0
dense_3693
dense_3695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_35952
dense/StatefulPartitionedCall?
5dense_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer1_3678*
_output_shapes

:
*
dtype027
5dense_layer1/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer1/kernel/Regularizer/SquareSquare=dense_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer1/kernel/Regularizer/Square?
%dense_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer1/kernel/Regularizer/Const?
#dense_layer1/kernel/Regularizer/SumSum*dense_layer1/kernel/Regularizer/Square:y:0.dense_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/Sum?
%dense_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer1/kernel/Regularizer/mul/x?
#dense_layer1/kernel/Regularizer/mulMul.dense_layer1/kernel/Regularizer/mul/x:output:0,dense_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/mul?
5dense_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer2_3683*
_output_shapes

:
*
dtype027
5dense_layer2/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer2/kernel/Regularizer/SquareSquare=dense_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer2/kernel/Regularizer/Square?
%dense_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer2/kernel/Regularizer/Const?
#dense_layer2/kernel/Regularizer/SumSum*dense_layer2/kernel/Regularizer/Square:y:0.dense_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/Sum?
%dense_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer2/kernel/Regularizer/mul/x?
#dense_layer2/kernel/Regularizer/mulMul.dense_layer2/kernel/Regularizer/mul/x:output:0,dense_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/mul?
5dense_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer3_3688*
_output_shapes

:
*
dtype027
5dense_layer3/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer3/kernel/Regularizer/SquareSquare=dense_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer3/kernel/Regularizer/Square?
%dense_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer3/kernel/Regularizer/Const?
#dense_layer3/kernel/Regularizer/SumSum*dense_layer3/kernel/Regularizer/Square:y:0.dense_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/Sum?
%dense_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer3/kernel/Regularizer/mul/x?
#dense_layer3/kernel/Regularizer/mulMul.dense_layer3/kernel/Regularizer/mul/x:output:0,dense_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/mul?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall%^dense_layer1/StatefulPartitionedCall%^dense_layer2/StatefulPartitionedCall%^dense_layer3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$dense_layer1/StatefulPartitionedCall$dense_layer1/StatefulPartitionedCall2L
$dense_layer2/StatefulPartitionedCall$dense_layer2/StatefulPartitionedCall2L
$dense_layer3/StatefulPartitionedCall$dense_layer3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?:
?
F__inference_functional_1_layer_call_and_return_conditional_losses_3898

inputs/
+dense_layer1_matmul_readvariableop_resource0
,dense_layer1_biasadd_readvariableop_resource/
+dense_layer2_matmul_readvariableop_resource0
,dense_layer2_biasadd_readvariableop_resource/
+dense_layer3_matmul_readvariableop_resource0
,dense_layer3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??
"dense_layer1/MatMul/ReadVariableOpReadVariableOp+dense_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"dense_layer1/MatMul/ReadVariableOp?
dense_layer1/MatMulMatMulinputs*dense_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_layer1/MatMul?
#dense_layer1/BiasAdd/ReadVariableOpReadVariableOp,dense_layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#dense_layer1/BiasAdd/ReadVariableOp?
dense_layer1/BiasAddBiasAdddense_layer1/MatMul:product:0+dense_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_layer1/BiasAdd
dense_layer1/ReluReludense_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_layer1/Relu?
"dense_layer2/MatMul/ReadVariableOpReadVariableOp+dense_layer2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"dense_layer2/MatMul/ReadVariableOp?
dense_layer2/MatMulMatMuldense_layer1/Relu:activations:0*dense_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer2/MatMul?
#dense_layer2/BiasAdd/ReadVariableOpReadVariableOp,dense_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#dense_layer2/BiasAdd/ReadVariableOp?
dense_layer2/BiasAddBiasAdddense_layer2/MatMul:product:0+dense_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer2/BiasAdd
dense_layer2/ReluReludense_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_layer2/Relu?
"dense_layer3/MatMul/ReadVariableOpReadVariableOp+dense_layer3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"dense_layer3/MatMul/ReadVariableOp?
dense_layer3/MatMulMatMuldense_layer2/Relu:activations:0*dense_layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_layer3/MatMul?
#dense_layer3/BiasAdd/ReadVariableOpReadVariableOp,dense_layer3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#dense_layer3/BiasAdd/ReadVariableOp?
dense_layer3/BiasAddBiasAdddense_layer3/MatMul:product:0+dense_layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_layer3/BiasAdd
dense_layer3/ReluReludense_layer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_layer3/Relu?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldense_layer3/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Softmax?
5dense_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+dense_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer1/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer1/kernel/Regularizer/SquareSquare=dense_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer1/kernel/Regularizer/Square?
%dense_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer1/kernel/Regularizer/Const?
#dense_layer1/kernel/Regularizer/SumSum*dense_layer1/kernel/Regularizer/Square:y:0.dense_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/Sum?
%dense_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer1/kernel/Regularizer/mul/x?
#dense_layer1/kernel/Regularizer/mulMul.dense_layer1/kernel/Regularizer/mul/x:output:0,dense_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/mul?
5dense_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+dense_layer2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer2/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer2/kernel/Regularizer/SquareSquare=dense_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer2/kernel/Regularizer/Square?
%dense_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer2/kernel/Regularizer/Const?
#dense_layer2/kernel/Regularizer/SumSum*dense_layer2/kernel/Regularizer/Square:y:0.dense_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/Sum?
%dense_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer2/kernel/Regularizer/mul/x?
#dense_layer2/kernel/Regularizer/mulMul.dense_layer2/kernel/Regularizer/mul/x:output:0,dense_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/mul?
5dense_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+dense_layer3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer3/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer3/kernel/Regularizer/SquareSquare=dense_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer3/kernel/Regularizer/Square?
%dense_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer3/kernel/Regularizer/Const?
#dense_layer3/kernel/Regularizer/SumSum*dense_layer3/kernel/Regularizer/Square:y:0.dense_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/Sum?
%dense_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer3/kernel/Regularizer/mul/x?
#dense_layer3/kernel/Regularizer/mulMul.dense_layer3/kernel/Regularizer/mul/x:output:0,dense_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/mulk
IdentityIdentitydense/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????:::::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_dense_layer3_layer_call_and_return_conditional_losses_4077

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
5dense_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer3/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer3/kernel/Regularizer/SquareSquare=dense_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer3/kernel/Regularizer/Square?
%dense_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer3/kernel/Regularizer/Const?
#dense_layer3/kernel/Regularizer/SumSum*dense_layer3/kernel/Regularizer/Square:y:0.dense_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/Sum?
%dense_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer3/kernel/Regularizer/mul/x?
#dense_layer3/kernel/Regularizer/mulMul.dense_layer3/kernel/Regularizer/mul/x:output:0,dense_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/mulf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
y
$__inference_dense_layer_call_fn_4106

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_35952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
+__inference_dense_layer2_layer_call_fn_4054

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer2_layer_call_and_return_conditional_losses_35352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
F__inference_dense_layer2_layer_call_and_return_conditional_losses_4045

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
5dense_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer2/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer2/kernel/Regularizer/SquareSquare=dense_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer2/kernel/Regularizer/Square?
%dense_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer2/kernel/Regularizer/Const?
#dense_layer2/kernel/Regularizer/SumSum*dense_layer2/kernel/Regularizer/Square:y:0.dense_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/Sum?
%dense_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer2/kernel/Regularizer/mul/x?
#dense_layer2/kernel/Regularizer/mulMul.dense_layer2/kernel/Regularizer/mul/x:output:0,dense_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/mulf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
q
__inference_loss_fn_1_4128B
>dense_layer2_kernel_regularizer_square_readvariableop_resource
identity??
5dense_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>dense_layer2_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer2/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer2/kernel/Regularizer/SquareSquare=dense_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer2/kernel/Regularizer/Square?
%dense_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer2/kernel/Regularizer/Const?
#dense_layer2/kernel/Regularizer/SumSum*dense_layer2/kernel/Regularizer/Square:y:0.dense_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/Sum?
%dense_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer2/kernel/Regularizer/mul/x?
#dense_layer2/kernel/Regularizer/mulMul.dense_layer2/kernel/Regularizer/mul/x:output:0,dense_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/mulj
IdentityIdentity'dense_layer2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?
?
+__inference_functional_1_layer_call_fn_3799	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_37802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
??
?
 __inference__traced_restore_4382
file_prefix(
$assignvariableop_dense_layer1_kernel(
$assignvariableop_1_dense_layer1_bias*
&assignvariableop_2_dense_layer2_kernel(
$assignvariableop_3_dense_layer2_bias*
&assignvariableop_4_dense_layer3_kernel(
$assignvariableop_5_dense_layer3_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1
assignvariableop_17_total_2
assignvariableop_18_count_22
.assignvariableop_19_adam_dense_layer1_kernel_m0
,assignvariableop_20_adam_dense_layer1_bias_m2
.assignvariableop_21_adam_dense_layer2_kernel_m0
,assignvariableop_22_adam_dense_layer2_bias_m2
.assignvariableop_23_adam_dense_layer3_kernel_m0
,assignvariableop_24_adam_dense_layer3_bias_m+
'assignvariableop_25_adam_dense_kernel_m)
%assignvariableop_26_adam_dense_bias_m2
.assignvariableop_27_adam_dense_layer1_kernel_v0
,assignvariableop_28_adam_dense_layer1_bias_v2
.assignvariableop_29_adam_dense_layer2_kernel_v0
,assignvariableop_30_adam_dense_layer2_bias_v2
.assignvariableop_31_adam_dense_layer3_kernel_v0
,assignvariableop_32_adam_dense_layer3_bias_v+
'assignvariableop_33_adam_dense_kernel_v)
%assignvariableop_34_adam_dense_bias_v
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_dense_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_dense_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_dense_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_dense_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_dense_layer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_dense_layer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_dense_layer1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_dense_layer1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_dense_layer2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_dense_layer2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_dense_layer3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_dense_layer3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_dense_layer1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_dense_layer1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_dense_layer2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_dense_layer2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_dense_layer3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_dense_layer3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_dense_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35?
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
+__inference_functional_1_layer_call_fn_3990

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_37802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_functional_1_layer_call_fn_3736	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_37172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
q
__inference_loss_fn_2_4139B
>dense_layer3_kernel_regularizer_square_readvariableop_resource
identity??
5dense_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>dense_layer3_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer3/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer3/kernel/Regularizer/SquareSquare=dense_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer3/kernel/Regularizer/Square?
%dense_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer3/kernel/Regularizer/Const?
#dense_layer3/kernel/Regularizer/SumSum*dense_layer3/kernel/Regularizer/Square:y:0.dense_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/Sum?
%dense_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer3/kernel/Regularizer/mul/x?
#dense_layer3/kernel/Regularizer/mulMul.dense_layer3/kernel/Regularizer/mul/x:output:0,dense_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/mulj
IdentityIdentity'dense_layer3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?'
?
__inference__wrapped_model_3481	
input<
8functional_1_dense_layer1_matmul_readvariableop_resource=
9functional_1_dense_layer1_biasadd_readvariableop_resource<
8functional_1_dense_layer2_matmul_readvariableop_resource=
9functional_1_dense_layer2_biasadd_readvariableop_resource<
8functional_1_dense_layer3_matmul_readvariableop_resource=
9functional_1_dense_layer3_biasadd_readvariableop_resource5
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource
identity??
/functional_1/dense_layer1/MatMul/ReadVariableOpReadVariableOp8functional_1_dense_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype021
/functional_1/dense_layer1/MatMul/ReadVariableOp?
 functional_1/dense_layer1/MatMulMatMulinput7functional_1/dense_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 functional_1/dense_layer1/MatMul?
0functional_1/dense_layer1/BiasAdd/ReadVariableOpReadVariableOp9functional_1_dense_layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype022
0functional_1/dense_layer1/BiasAdd/ReadVariableOp?
!functional_1/dense_layer1/BiasAddBiasAdd*functional_1/dense_layer1/MatMul:product:08functional_1/dense_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2#
!functional_1/dense_layer1/BiasAdd?
functional_1/dense_layer1/ReluRelu*functional_1/dense_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2 
functional_1/dense_layer1/Relu?
/functional_1/dense_layer2/MatMul/ReadVariableOpReadVariableOp8functional_1_dense_layer2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype021
/functional_1/dense_layer2/MatMul/ReadVariableOp?
 functional_1/dense_layer2/MatMulMatMul,functional_1/dense_layer1/Relu:activations:07functional_1/dense_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 functional_1/dense_layer2/MatMul?
0functional_1/dense_layer2/BiasAdd/ReadVariableOpReadVariableOp9functional_1_dense_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0functional_1/dense_layer2/BiasAdd/ReadVariableOp?
!functional_1/dense_layer2/BiasAddBiasAdd*functional_1/dense_layer2/MatMul:product:08functional_1/dense_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!functional_1/dense_layer2/BiasAdd?
functional_1/dense_layer2/ReluRelu*functional_1/dense_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
functional_1/dense_layer2/Relu?
/functional_1/dense_layer3/MatMul/ReadVariableOpReadVariableOp8functional_1_dense_layer3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype021
/functional_1/dense_layer3/MatMul/ReadVariableOp?
 functional_1/dense_layer3/MatMulMatMul,functional_1/dense_layer2/Relu:activations:07functional_1/dense_layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 functional_1/dense_layer3/MatMul?
0functional_1/dense_layer3/BiasAdd/ReadVariableOpReadVariableOp9functional_1_dense_layer3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype022
0functional_1/dense_layer3/BiasAdd/ReadVariableOp?
!functional_1/dense_layer3/BiasAddBiasAdd*functional_1/dense_layer3/MatMul:product:08functional_1/dense_layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2#
!functional_1/dense_layer3/BiasAdd?
functional_1/dense_layer3/ReluRelu*functional_1/dense_layer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2 
functional_1/dense_layer3/Relu?
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02*
(functional_1/dense/MatMul/ReadVariableOp?
functional_1/dense/MatMulMatMul,functional_1/dense_layer3/Relu:activations:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_1/dense/MatMul?
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOp?
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_1/dense/BiasAdd?
functional_1/dense/SoftmaxSoftmax#functional_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
functional_1/dense/Softmaxx
IdentityIdentity$functional_1/dense/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????:::::::::N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
F__inference_dense_layer1_layer_call_and_return_conditional_losses_3502

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
5dense_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer1/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer1/kernel/Regularizer/SquareSquare=dense_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer1/kernel/Regularizer/Square?
%dense_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer1/kernel/Regularizer/Const?
#dense_layer1/kernel/Regularizer/SumSum*dense_layer1/kernel/Regularizer/Square:y:0.dense_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/Sum?
%dense_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer1/kernel/Regularizer/mul/x?
#dense_layer1/kernel/Regularizer/mulMul.dense_layer1/kernel/Regularizer/mul/x:output:0,dense_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/mulf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
F__inference_functional_1_layer_call_and_return_conditional_losses_3672	
input
dense_layer1_3633
dense_layer1_3635
dense_layer2_3638
dense_layer2_3640
dense_layer3_3643
dense_layer3_3645

dense_3648

dense_3650
identity??dense/StatefulPartitionedCall?$dense_layer1/StatefulPartitionedCall?$dense_layer2/StatefulPartitionedCall?$dense_layer3/StatefulPartitionedCall?
$dense_layer1/StatefulPartitionedCallStatefulPartitionedCallinputdense_layer1_3633dense_layer1_3635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer1_layer_call_and_return_conditional_losses_35022&
$dense_layer1/StatefulPartitionedCall?
$dense_layer2/StatefulPartitionedCallStatefulPartitionedCall-dense_layer1/StatefulPartitionedCall:output:0dense_layer2_3638dense_layer2_3640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer2_layer_call_and_return_conditional_losses_35352&
$dense_layer2/StatefulPartitionedCall?
$dense_layer3/StatefulPartitionedCallStatefulPartitionedCall-dense_layer2/StatefulPartitionedCall:output:0dense_layer3_3643dense_layer3_3645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer3_layer_call_and_return_conditional_losses_35682&
$dense_layer3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-dense_layer3/StatefulPartitionedCall:output:0
dense_3648
dense_3650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_35952
dense/StatefulPartitionedCall?
5dense_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer1_3633*
_output_shapes

:
*
dtype027
5dense_layer1/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer1/kernel/Regularizer/SquareSquare=dense_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer1/kernel/Regularizer/Square?
%dense_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer1/kernel/Regularizer/Const?
#dense_layer1/kernel/Regularizer/SumSum*dense_layer1/kernel/Regularizer/Square:y:0.dense_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/Sum?
%dense_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer1/kernel/Regularizer/mul/x?
#dense_layer1/kernel/Regularizer/mulMul.dense_layer1/kernel/Regularizer/mul/x:output:0,dense_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/mul?
5dense_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer2_3638*
_output_shapes

:
*
dtype027
5dense_layer2/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer2/kernel/Regularizer/SquareSquare=dense_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer2/kernel/Regularizer/Square?
%dense_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer2/kernel/Regularizer/Const?
#dense_layer2/kernel/Regularizer/SumSum*dense_layer2/kernel/Regularizer/Square:y:0.dense_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/Sum?
%dense_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer2/kernel/Regularizer/mul/x?
#dense_layer2/kernel/Regularizer/mulMul.dense_layer2/kernel/Regularizer/mul/x:output:0,dense_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/mul?
5dense_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer3_3643*
_output_shapes

:
*
dtype027
5dense_layer3/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer3/kernel/Regularizer/SquareSquare=dense_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer3/kernel/Regularizer/Square?
%dense_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer3/kernel/Regularizer/Const?
#dense_layer3/kernel/Regularizer/SumSum*dense_layer3/kernel/Regularizer/Square:y:0.dense_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/Sum?
%dense_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer3/kernel/Regularizer/mul/x?
#dense_layer3/kernel/Regularizer/mulMul.dense_layer3/kernel/Regularizer/mul/x:output:0,dense_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/mul?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall%^dense_layer1/StatefulPartitionedCall%^dense_layer2/StatefulPartitionedCall%^dense_layer3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$dense_layer1/StatefulPartitionedCall$dense_layer1/StatefulPartitionedCall2L
$dense_layer2/StatefulPartitionedCall$dense_layer2/StatefulPartitionedCall2L
$dense_layer3/StatefulPartitionedCall$dense_layer3/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
+__inference_dense_layer1_layer_call_fn_4022

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer1_layer_call_and_return_conditional_losses_35022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_3848	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_34812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
q
__inference_loss_fn_0_4117B
>dense_layer1_kernel_regularizer_square_readvariableop_resource
identity??
5dense_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>dense_layer1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer1/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer1/kernel/Regularizer/SquareSquare=dense_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer1/kernel/Regularizer/Square?
%dense_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer1/kernel/Regularizer/Const?
#dense_layer1/kernel/Regularizer/SumSum*dense_layer1/kernel/Regularizer/Square:y:0.dense_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/Sum?
%dense_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer1/kernel/Regularizer/mul/x?
#dense_layer1/kernel/Regularizer/mulMul.dense_layer1/kernel/Regularizer/mul/x:output:0,dense_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/mulj
IdentityIdentity'dense_layer1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?
?
F__inference_dense_layer2_layer_call_and_return_conditional_losses_3535

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
5dense_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer2/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer2/kernel/Regularizer/SquareSquare=dense_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer2/kernel/Regularizer/Square?
%dense_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer2/kernel/Regularizer/Const?
#dense_layer2/kernel/Regularizer/SumSum*dense_layer2/kernel/Regularizer/Square:y:0.dense_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/Sum?
%dense_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer2/kernel/Regularizer/mul/x?
#dense_layer2/kernel/Regularizer/mulMul.dense_layer2/kernel/Regularizer/mul/x:output:0,dense_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/mulf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?:
?
F__inference_functional_1_layer_call_and_return_conditional_losses_3948

inputs/
+dense_layer1_matmul_readvariableop_resource0
,dense_layer1_biasadd_readvariableop_resource/
+dense_layer2_matmul_readvariableop_resource0
,dense_layer2_biasadd_readvariableop_resource/
+dense_layer3_matmul_readvariableop_resource0
,dense_layer3_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??
"dense_layer1/MatMul/ReadVariableOpReadVariableOp+dense_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"dense_layer1/MatMul/ReadVariableOp?
dense_layer1/MatMulMatMulinputs*dense_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_layer1/MatMul?
#dense_layer1/BiasAdd/ReadVariableOpReadVariableOp,dense_layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#dense_layer1/BiasAdd/ReadVariableOp?
dense_layer1/BiasAddBiasAdddense_layer1/MatMul:product:0+dense_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_layer1/BiasAdd
dense_layer1/ReluReludense_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_layer1/Relu?
"dense_layer2/MatMul/ReadVariableOpReadVariableOp+dense_layer2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"dense_layer2/MatMul/ReadVariableOp?
dense_layer2/MatMulMatMuldense_layer1/Relu:activations:0*dense_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer2/MatMul?
#dense_layer2/BiasAdd/ReadVariableOpReadVariableOp,dense_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#dense_layer2/BiasAdd/ReadVariableOp?
dense_layer2/BiasAddBiasAdddense_layer2/MatMul:product:0+dense_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer2/BiasAdd
dense_layer2/ReluReludense_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_layer2/Relu?
"dense_layer3/MatMul/ReadVariableOpReadVariableOp+dense_layer3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"dense_layer3/MatMul/ReadVariableOp?
dense_layer3/MatMulMatMuldense_layer2/Relu:activations:0*dense_layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_layer3/MatMul?
#dense_layer3/BiasAdd/ReadVariableOpReadVariableOp,dense_layer3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#dense_layer3/BiasAdd/ReadVariableOp?
dense_layer3/BiasAddBiasAdddense_layer3/MatMul:product:0+dense_layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_layer3/BiasAdd
dense_layer3/ReluReludense_layer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_layer3/Relu?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldense_layer3/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Softmax?
5dense_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+dense_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer1/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer1/kernel/Regularizer/SquareSquare=dense_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer1/kernel/Regularizer/Square?
%dense_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer1/kernel/Regularizer/Const?
#dense_layer1/kernel/Regularizer/SumSum*dense_layer1/kernel/Regularizer/Square:y:0.dense_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/Sum?
%dense_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer1/kernel/Regularizer/mul/x?
#dense_layer1/kernel/Regularizer/mulMul.dense_layer1/kernel/Regularizer/mul/x:output:0,dense_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/mul?
5dense_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+dense_layer2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer2/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer2/kernel/Regularizer/SquareSquare=dense_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer2/kernel/Regularizer/Square?
%dense_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer2/kernel/Regularizer/Const?
#dense_layer2/kernel/Regularizer/SumSum*dense_layer2/kernel/Regularizer/Square:y:0.dense_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/Sum?
%dense_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer2/kernel/Regularizer/mul/x?
#dense_layer2/kernel/Regularizer/mulMul.dense_layer2/kernel/Regularizer/mul/x:output:0,dense_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/mul?
5dense_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+dense_layer3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer3/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer3/kernel/Regularizer/SquareSquare=dense_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer3/kernel/Regularizer/Square?
%dense_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer3/kernel/Regularizer/Const?
#dense_layer3/kernel/Regularizer/SumSum*dense_layer3/kernel/Regularizer/Square:y:0.dense_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/Sum?
%dense_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer3/kernel/Regularizer/mul/x?
#dense_layer3/kernel/Regularizer/mulMul.dense_layer3/kernel/Regularizer/mul/x:output:0,dense_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/mulk
IdentityIdentitydense/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????:::::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
F__inference_functional_1_layer_call_and_return_conditional_losses_3630	
input
dense_layer1_3513
dense_layer1_3515
dense_layer2_3546
dense_layer2_3548
dense_layer3_3579
dense_layer3_3581

dense_3606

dense_3608
identity??dense/StatefulPartitionedCall?$dense_layer1/StatefulPartitionedCall?$dense_layer2/StatefulPartitionedCall?$dense_layer3/StatefulPartitionedCall?
$dense_layer1/StatefulPartitionedCallStatefulPartitionedCallinputdense_layer1_3513dense_layer1_3515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer1_layer_call_and_return_conditional_losses_35022&
$dense_layer1/StatefulPartitionedCall?
$dense_layer2/StatefulPartitionedCallStatefulPartitionedCall-dense_layer1/StatefulPartitionedCall:output:0dense_layer2_3546dense_layer2_3548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer2_layer_call_and_return_conditional_losses_35352&
$dense_layer2/StatefulPartitionedCall?
$dense_layer3/StatefulPartitionedCallStatefulPartitionedCall-dense_layer2/StatefulPartitionedCall:output:0dense_layer3_3579dense_layer3_3581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer3_layer_call_and_return_conditional_losses_35682&
$dense_layer3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-dense_layer3/StatefulPartitionedCall:output:0
dense_3606
dense_3608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_35952
dense/StatefulPartitionedCall?
5dense_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer1_3513*
_output_shapes

:
*
dtype027
5dense_layer1/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer1/kernel/Regularizer/SquareSquare=dense_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer1/kernel/Regularizer/Square?
%dense_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer1/kernel/Regularizer/Const?
#dense_layer1/kernel/Regularizer/SumSum*dense_layer1/kernel/Regularizer/Square:y:0.dense_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/Sum?
%dense_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer1/kernel/Regularizer/mul/x?
#dense_layer1/kernel/Regularizer/mulMul.dense_layer1/kernel/Regularizer/mul/x:output:0,dense_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/mul?
5dense_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer2_3546*
_output_shapes

:
*
dtype027
5dense_layer2/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer2/kernel/Regularizer/SquareSquare=dense_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer2/kernel/Regularizer/Square?
%dense_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer2/kernel/Regularizer/Const?
#dense_layer2/kernel/Regularizer/SumSum*dense_layer2/kernel/Regularizer/Square:y:0.dense_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/Sum?
%dense_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer2/kernel/Regularizer/mul/x?
#dense_layer2/kernel/Regularizer/mulMul.dense_layer2/kernel/Regularizer/mul/x:output:0,dense_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/mul?
5dense_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer3_3579*
_output_shapes

:
*
dtype027
5dense_layer3/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer3/kernel/Regularizer/SquareSquare=dense_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer3/kernel/Regularizer/Square?
%dense_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer3/kernel/Regularizer/Const?
#dense_layer3/kernel/Regularizer/SumSum*dense_layer3/kernel/Regularizer/Square:y:0.dense_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/Sum?
%dense_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer3/kernel/Regularizer/mul/x?
#dense_layer3/kernel/Regularizer/mulMul.dense_layer3/kernel/Regularizer/mul/x:output:0,dense_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/mul?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall%^dense_layer1/StatefulPartitionedCall%^dense_layer2/StatefulPartitionedCall%^dense_layer3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$dense_layer1/StatefulPartitionedCall$dense_layer1/StatefulPartitionedCall2L
$dense_layer2/StatefulPartitionedCall$dense_layer2/StatefulPartitionedCall2L
$dense_layer3/StatefulPartitionedCall$dense_layer3/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
F__inference_dense_layer1_layer_call_and_return_conditional_losses_4013

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
5dense_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5dense_layer1/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer1/kernel/Regularizer/SquareSquare=dense_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer1/kernel/Regularizer/Square?
%dense_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer1/kernel/Regularizer/Const?
#dense_layer1/kernel/Regularizer/SumSum*dense_layer1/kernel/Regularizer/Square:y:0.dense_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/Sum?
%dense_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer1/kernel/Regularizer/mul/x?
#dense_layer1/kernel/Regularizer/mulMul.dense_layer1/kernel/Regularizer/mul/x:output:0,dense_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/mulf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_layer3_layer_call_fn_4086

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer3_layer_call_and_return_conditional_losses_35682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
?__inference_dense_layer_call_and_return_conditional_losses_3595

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?1
?
F__inference_functional_1_layer_call_and_return_conditional_losses_3780

inputs
dense_layer1_3741
dense_layer1_3743
dense_layer2_3746
dense_layer2_3748
dense_layer3_3751
dense_layer3_3753

dense_3756

dense_3758
identity??dense/StatefulPartitionedCall?$dense_layer1/StatefulPartitionedCall?$dense_layer2/StatefulPartitionedCall?$dense_layer3/StatefulPartitionedCall?
$dense_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_layer1_3741dense_layer1_3743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer1_layer_call_and_return_conditional_losses_35022&
$dense_layer1/StatefulPartitionedCall?
$dense_layer2/StatefulPartitionedCallStatefulPartitionedCall-dense_layer1/StatefulPartitionedCall:output:0dense_layer2_3746dense_layer2_3748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer2_layer_call_and_return_conditional_losses_35352&
$dense_layer2/StatefulPartitionedCall?
$dense_layer3/StatefulPartitionedCallStatefulPartitionedCall-dense_layer2/StatefulPartitionedCall:output:0dense_layer3_3751dense_layer3_3753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_layer3_layer_call_and_return_conditional_losses_35682&
$dense_layer3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-dense_layer3/StatefulPartitionedCall:output:0
dense_3756
dense_3758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_35952
dense/StatefulPartitionedCall?
5dense_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer1_3741*
_output_shapes

:
*
dtype027
5dense_layer1/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer1/kernel/Regularizer/SquareSquare=dense_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer1/kernel/Regularizer/Square?
%dense_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer1/kernel/Regularizer/Const?
#dense_layer1/kernel/Regularizer/SumSum*dense_layer1/kernel/Regularizer/Square:y:0.dense_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/Sum?
%dense_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer1/kernel/Regularizer/mul/x?
#dense_layer1/kernel/Regularizer/mulMul.dense_layer1/kernel/Regularizer/mul/x:output:0,dense_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer1/kernel/Regularizer/mul?
5dense_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer2_3746*
_output_shapes

:
*
dtype027
5dense_layer2/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer2/kernel/Regularizer/SquareSquare=dense_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer2/kernel/Regularizer/Square?
%dense_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer2/kernel/Regularizer/Const?
#dense_layer2/kernel/Regularizer/SumSum*dense_layer2/kernel/Regularizer/Square:y:0.dense_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/Sum?
%dense_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer2/kernel/Regularizer/mul/x?
#dense_layer2/kernel/Regularizer/mulMul.dense_layer2/kernel/Regularizer/mul/x:output:0,dense_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer2/kernel/Regularizer/mul?
5dense_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_layer3_3751*
_output_shapes

:
*
dtype027
5dense_layer3/kernel/Regularizer/Square/ReadVariableOp?
&dense_layer3/kernel/Regularizer/SquareSquare=dense_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
2(
&dense_layer3/kernel/Regularizer/Square?
%dense_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_layer3/kernel/Regularizer/Const?
#dense_layer3/kernel/Regularizer/SumSum*dense_layer3/kernel/Regularizer/Square:y:0.dense_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/Sum?
%dense_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%dense_layer3/kernel/Regularizer/mul/x?
#dense_layer3/kernel/Regularizer/mulMul.dense_layer3/kernel/Regularizer/mul/x:output:0,dense_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_layer3/kernel/Regularizer/mul?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall%^dense_layer1/StatefulPartitionedCall%^dense_layer2/StatefulPartitionedCall%^dense_layer3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$dense_layer1/StatefulPartitionedCall$dense_layer1/StatefulPartitionedCall2L
$dense_layer2/StatefulPartitionedCall$dense_layer2/StatefulPartitionedCall2L
$dense_layer3/StatefulPartitionedCall$dense_layer3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_functional_1_layer_call_fn_3969

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_37172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?K
?
__inference__traced_save_4267
file_prefix2
.savev2_dense_layer1_kernel_read_readvariableop0
,savev2_dense_layer1_bias_read_readvariableop2
.savev2_dense_layer2_kernel_read_readvariableop0
,savev2_dense_layer2_bias_read_readvariableop2
.savev2_dense_layer3_kernel_read_readvariableop0
,savev2_dense_layer3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop9
5savev2_adam_dense_layer1_kernel_m_read_readvariableop7
3savev2_adam_dense_layer1_bias_m_read_readvariableop9
5savev2_adam_dense_layer2_kernel_m_read_readvariableop7
3savev2_adam_dense_layer2_bias_m_read_readvariableop9
5savev2_adam_dense_layer3_kernel_m_read_readvariableop7
3savev2_adam_dense_layer3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop9
5savev2_adam_dense_layer1_kernel_v_read_readvariableop7
3savev2_adam_dense_layer1_bias_v_read_readvariableop9
5savev2_adam_dense_layer2_kernel_v_read_readvariableop7
3savev2_adam_dense_layer2_bias_v_read_readvariableop9
5savev2_adam_dense_layer3_kernel_v_read_readvariableop7
3savev2_adam_dense_layer3_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_234745a1f1414ae9a7338f75348db9a8/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_dense_layer1_kernel_read_readvariableop,savev2_dense_layer1_bias_read_readvariableop.savev2_dense_layer2_kernel_read_readvariableop,savev2_dense_layer2_bias_read_readvariableop.savev2_dense_layer3_kernel_read_readvariableop,savev2_dense_layer3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop5savev2_adam_dense_layer1_kernel_m_read_readvariableop3savev2_adam_dense_layer1_bias_m_read_readvariableop5savev2_adam_dense_layer2_kernel_m_read_readvariableop3savev2_adam_dense_layer2_bias_m_read_readvariableop5savev2_adam_dense_layer3_kernel_m_read_readvariableop3savev2_adam_dense_layer3_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop5savev2_adam_dense_layer1_kernel_v_read_readvariableop3savev2_adam_dense_layer1_bias_v_read_readvariableop5savev2_adam_dense_layer2_kernel_v_read_readvariableop3savev2_adam_dense_layer2_bias_v_read_readvariableop5savev2_adam_dense_layer3_kernel_v_read_readvariableop3savev2_adam_dense_layer3_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
:
:
::
:
:
:: : : : : : : : : : : :
:
:
::
:
:
::
:
:
::
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$  

_output_shapes

:
: !

_output_shapes
:
:$" 

_output_shapes

:
: #

_output_shapes
::$

_output_shapes
: 
?
?
?__inference_dense_layer_call_and_return_conditional_losses_4097

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input.
serving_default_input:0?????????9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?0
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
c__call__
d_default_save_signature
*e&call_and_return_all_conditional_losses"?-
_tf_keras_network?,{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_layer1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer1", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_layer2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer2", "inbound_nodes": [[["dense_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_layer3", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer3", "inbound_nodes": [[["dense_layer2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dense_layer3", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_layer1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer1", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_layer2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer2", "inbound_nodes": [[["dense_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_layer3", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer3", "inbound_nodes": [[["dense_layer2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dense_layer3", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy", "binary_accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
f__call__
*g&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_layer1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h__call__
*i&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_layer2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_layer3", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
l__call__
*m&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?
$iter

%beta_1

&beta_2
	'decay
(learning_ratemSmTmUmVmWmXmYmZv[v\v]v^v_v`vavb"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
5
n0
o1
p2"
trackable_list_wrapper
?
)non_trainable_variables
*layer_metrics
+metrics
trainable_variables

,layers
	variables
-layer_regularization_losses
	regularization_losses
c__call__
d_default_save_signature
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
,
qserving_default"
signature_map
%:#
2dense_layer1/kernel
:
2dense_layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
n0"
trackable_list_wrapper
?
.non_trainable_variables
/layer_metrics
0metrics
trainable_variables

1layers
	variables
2layer_regularization_losses
regularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
%:#
2dense_layer2/kernel
:2dense_layer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
o0"
trackable_list_wrapper
?
3non_trainable_variables
4layer_metrics
5metrics
trainable_variables

6layers
	variables
7layer_regularization_losses
regularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
%:#
2dense_layer3/kernel
:
2dense_layer3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
p0"
trackable_list_wrapper
?
8non_trainable_variables
9layer_metrics
:metrics
trainable_variables

;layers
	variables
<layer_regularization_losses
regularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
:
2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
=non_trainable_variables
>layer_metrics
?metrics
 trainable_variables

@layers
!	variables
Alayer_regularization_losses
"regularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
B0
C1
D2"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
n0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
p0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Etotal
	Fcount
G	variables
H	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	Itotal
	Jcount
K
_fn_kwargs
L	variables
M	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
?
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "binary_accuracy", "dtype": "float32", "config": {"name": "binary_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
.
E0
F1"
trackable_list_wrapper
-
G	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
*:(
2Adam/dense_layer1/kernel/m
$:"
2Adam/dense_layer1/bias/m
*:(
2Adam/dense_layer2/kernel/m
$:"2Adam/dense_layer2/bias/m
*:(
2Adam/dense_layer3/kernel/m
$:"
2Adam/dense_layer3/bias/m
#:!
2Adam/dense/kernel/m
:2Adam/dense/bias/m
*:(
2Adam/dense_layer1/kernel/v
$:"
2Adam/dense_layer1/bias/v
*:(
2Adam/dense_layer2/kernel/v
$:"2Adam/dense_layer2/bias/v
*:(
2Adam/dense_layer3/kernel/v
$:"
2Adam/dense_layer3/bias/v
#:!
2Adam/dense/kernel/v
:2Adam/dense/bias/v
?2?
+__inference_functional_1_layer_call_fn_3799
+__inference_functional_1_layer_call_fn_3736
+__inference_functional_1_layer_call_fn_3969
+__inference_functional_1_layer_call_fn_3990?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_3481?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *$?!
?
input?????????
?2?
F__inference_functional_1_layer_call_and_return_conditional_losses_3630
F__inference_functional_1_layer_call_and_return_conditional_losses_3898
F__inference_functional_1_layer_call_and_return_conditional_losses_3948
F__inference_functional_1_layer_call_and_return_conditional_losses_3672?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dense_layer1_layer_call_fn_4022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_layer1_layer_call_and_return_conditional_losses_4013?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_layer2_layer_call_fn_4054?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_layer2_layer_call_and_return_conditional_losses_4045?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_layer3_layer_call_fn_4086?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_layer3_layer_call_and_return_conditional_losses_4077?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_4106?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_4097?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_4117?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_4128?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_4139?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
/B-
"__inference_signature_wrapper_3848input?
__inference__wrapped_model_3481i.?+
$?!
?
input?????????
? "-?*
(
dense?
dense??????????
F__inference_dense_layer1_layer_call_and_return_conditional_losses_4013\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? ~
+__inference_dense_layer1_layer_call_fn_4022O/?,
%?"
 ?
inputs?????????
? "??????????
?
F__inference_dense_layer2_layer_call_and_return_conditional_losses_4045\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ~
+__inference_dense_layer2_layer_call_fn_4054O/?,
%?"
 ?
inputs?????????

? "???????????
F__inference_dense_layer3_layer_call_and_return_conditional_losses_4077\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? ~
+__inference_dense_layer3_layer_call_fn_4086O/?,
%?"
 ?
inputs?????????
? "??????????
?
?__inference_dense_layer_call_and_return_conditional_losses_4097\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? w
$__inference_dense_layer_call_fn_4106O/?,
%?"
 ?
inputs?????????

? "???????????
F__inference_functional_1_layer_call_and_return_conditional_losses_3630i6?3
,?)
?
input?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_functional_1_layer_call_and_return_conditional_losses_3672i6?3
,?)
?
input?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_functional_1_layer_call_and_return_conditional_losses_3898j7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_functional_1_layer_call_and_return_conditional_losses_3948j7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
+__inference_functional_1_layer_call_fn_3736\6?3
,?)
?
input?????????
p

 
? "???????????
+__inference_functional_1_layer_call_fn_3799\6?3
,?)
?
input?????????
p 

 
? "???????????
+__inference_functional_1_layer_call_fn_3969]7?4
-?*
 ?
inputs?????????
p

 
? "???????????
+__inference_functional_1_layer_call_fn_3990]7?4
-?*
 ?
inputs?????????
p 

 
? "??????????9
__inference_loss_fn_0_4117?

? 
? "? 9
__inference_loss_fn_1_4128?

? 
? "? 9
__inference_loss_fn_2_4139?

? 
? "? ?
"__inference_signature_wrapper_3848r7?4
? 
-?*
(
input?
input?????????"-?*
(
dense?
dense?????????