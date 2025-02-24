#!/bin/zsh
scorefile=$2
# mkdir -p fastas
touch $scorefile
printf 'name\t-logP_Struc\t-logP_Contact\tdist_sc\tcon_sc\tarea_sc\tper_con\tper_no_con\tcontact_order\n' >> $scorefile
# printf 'name\tangles\n' >> $scorefile
for i in "$1/"*.pdb
do
	x=${i%.*}
	python get_seq.py $x.pdb >> $x.fa
	tr_rosetta $x.fa
	python -W ignore score.py -p $x.pdb -n $x.npz -s $scorefile
	# python -W ignore angles.py -p $x.pdb -n $x.npz -s $scorefile
done
