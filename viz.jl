### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ cd33cc66-1df7-11ec-102a-d5816d93431c
begin
	using CSV
	using DataFrames
	using Plots
	using Statistics
	using Loess
	using Interpolations
	using HypothesisTests
end

# ╔═╡ b29d2430-3950-4544-8daf-57be549a690f
ct = 10

# ╔═╡ 1b4dc0db-9f47-4a94-9bbc-b70c30972997
TR = "Test Reward"

# ╔═╡ 006035ef-08f6-4f9f-9dfb-043abfaf6af6
CR = "Mean Controller Reward"

# ╔═╡ c1dae931-ab39-4d98-bb9d-0453ce0e9e52
PR = "Mean Primitive Reward"

# ╔═╡ a36250ad-7318-4e57-a6ae-c36e782d2214
TIME = "Time (h)"

# ╔═╡ 7df4d38a-4b25-437c-975c-78010c3729c1
SAMPLES = "Samples"

# ╔═╡ bdbd896d-0761-4d30-82ab-928889953d3f
pathprefix = "../ScalableHrlEs/csv_logs"

# ╔═╡ 3163445d-ce5e-424a-961a-5ca03a34f177
begin
	# Gather
	gather_nohot_path = "$pathprefix/gather/tuned/nohot/AntGather_tuned_"
	gather_onehot_path = "$pathprefix/gather/tuned/onehot/AntGather_tuned-onehot_"
	gather_pt_path = "$pathprefix/gather/tl/AntGather-pre_pt1_"
	gather_ns_path = "$pathprefix/gather/ns/AntGather-ns-endposbehv_ns_"
	gather_5_path = "$pathprefix/gather/speedup/AntGather_5node_"
	gather_10_path = "$pathprefix/gather/speedup/AntGather_10node_"
	gather_single_path = "$pathprefix/gather/AntGather_single_"

	# Maze
	maze_path = "$pathprefix/maze/tuned/AntMaze_tuned_"
	maze_pt_path = "$pathprefix/maze/tl/AntMaze-pre-fix_pt_"
	maze_ns_path = "$pathprefix/maze/ns/AntMaze-ns-endposbehv_ns_"
	maze_5_node_path = "$pathprefix/maze/speedup/AntMaze_5node_"
	maze_10_node_path = "$pathprefix/maze/speedup/AntMaze_10node_"
	maze_single_path = "$pathprefix/maze/AntMaze_single_"


	# Push
	push_path = "$pathprefix/push/tuned/AntPush_tuned_"
	push_pt_path = "$pathprefix/push/tl/AntPush-pre_pt1_"
	push_ns_path = "$pathprefix/push/ns/AntPush-ns-endposbehv_ns_"
	push_5_node_path = "$pathprefix/push/speedup/AntPush_5node_"
	push_10_node_path = "$pathprefix/push/speedup/AntPush_10node_"
	push_single_path = "$pathprefix/push/AntPush_single_"

end

# ╔═╡ 27ea31d5-52ab-4186-8859-48e912abbe07
begin
	gthr_tune_base_pth = "$pathprefix/gather/tune/AntGather_base_"
	
	gthr_tune_cd2_pth = "$pathprefix/gather/tune/AntGather_tune_cdist-2_"
	gthr_tune_cd8_pth = "$pathprefix/gather/tune/AntGather_tune_cdist-8_"

	gthr_tune_ep3_pth = "$pathprefix/gather/tune/AntGather_tune_eps-3_"
	gthr_tune_ep10_pth = "$pathprefix/gather/tune/AntGather_tune_eps-10_"

	gthr_tune_int10_pth = "$pathprefix/gather/tune/AntGather_tune_interval-10_"
	gthr_tune_int100_pth = "$pathprefix/gather/tune/AntGather_tune_interval-100_"

	gthr_tune_lr01_pth = "$pathprefix/gather/tune/AntGather_tune_lr-0.1_"
	gthr_tune_lr0001_pth = "$pathprefix/gather/tune/AntGather_tune_lr-0.001_"

	gthr_tune_ppg512_pth = "$pathprefix/gather/tune/AntGather_tune_ppg-512_"
	gthr_tune_ppg1000_pth = "$pathprefix/gather/tune/AntGather_tune_ppg-1000_"

	gthr_tune_sigma02_pth = "$pathprefix/gather/tune/AntGather_tune_sigma-0.2_"
	gthr_tune_sigma0002_pth = "$pathprefix/gather/tune/AntGather_tune_sigma-0.002_"
end

# ╔═╡ 1341a522-35b1-4d12-acc5-5455cff02992
begin
	mze_tune_base_pth = "$pathprefix/maze/tune/AntMaze_base_"
	
	mze_tune_cd2_pth = "$pathprefix/maze/tune/AntMaze_tune_cdist-2_"
	mze_tune_cd8_pth = "$pathprefix/maze/tune/AntMaze_tune_cdist-8_"

	mze_tune_ep3_pth = "$pathprefix/maze/tune/AntMaze_tune_eps-3_"
	mze_tune_ep10_pth = "$pathprefix/maze/tune/AntMaze_tune_eps-10_"

	mze_tune_int10_pth = "$pathprefix/maze/tune/AntMaze_tune_interval-10_"
	mze_tune_int100_pth = "$pathprefix/maze/tune/AntMaze_tune_interval-100_"

	mze_tune_lr01_pth = "$pathprefix/maze/tune/AntMaze_tune_lr-0.1_"
	mze_tune_lr0001_pth = "$pathprefix/maze/tune/AntMaze_tune_lr-0.001_"

	mze_tune_ppg512_pth = "$pathprefix/maze/tune/AntMaze_tune_ppg-512_"
	mze_tune_ppg1000_pth = "$pathprefix/maze/tune/AntMaze_tune_ppg-1000_"

	mze_tune_sigma02_pth = "$pathprefix/maze/tune/AntMaze_tune_sigma-0.2_"
	mze_tune_sigma0002_pth = "$pathprefix/maze/tune/AntMaze_tune_sigma-0.002_"
end

# ╔═╡ a28bbacf-5fa1-4a60-9ac5-66ca96dd0765
begin
	psh_tune_base_pth = "$pathprefix/push/tune/AntPush_base_"
	
	psh_tune_cd2_pth = "$pathprefix/push/tune/AntPush_tune_cdist-2_"
	psh_tune_cd8_pth = "$pathprefix/push/tune/AntPush_tune_cdist-8_"

	psh_tune_ep3_pth = "$pathprefix/push/tune/AntPush_tune_eps-3_"
	psh_tune_ep10_pth = "$pathprefix/push/tune/AntPush_tune_eps-10_"

	psh_tune_int10_pth = "$pathprefix/push/tune/AntPush_tune_interval-10_"
	psh_tune_int100_pth = "$pathprefix/push/tune/AntPush_tune_interval-100_"

	psh_tune_lr01_pth = "$pathprefix/push/tune/AntPush_tune_lr-0.1_"
	psh_tune_lr0001_pth = "$pathprefix/push/tune/AntPush_tune_lr-0.001_"

	psh_tune_ppg512_pth = "$pathprefix/push/tune/AntPush_tune_ppg-512_"
	psh_tune_ppg1000_pth = "$pathprefix/push/tune/AntPush_tune_ppg-1000_"

	psh_tune_sigma02_pth = "$pathprefix/push/tune/AntPush_tune_sigma-0.2_"
	psh_tune_sigma0002_pth = "$pathprefix/push/tune/AntPush_tune_sigma-0.002_"
end

# ╔═╡ 4bab0152-d53f-4bb9-a38a-49930797c632
begin 
	hiro_maze_eval_pth = "$pathprefix/hiro/maze/eval1.csv"
	hiro_maze_train_pth = "$pathprefix/hiro/maze/train1.csv"
end

# ╔═╡ 35dc88a2-9b9a-4d57-8036-ca198020276c
struct TuneRun
	base
	cd2
	cd8
	ep3
	ep10
	int10
	int100
	lr01
	lr0001
	ppg512
	ppg1000
	sigma02
	sigma0002
end

# ╔═╡ 23463959-ea89-4276-80a1-92cb3ff105b0
function fix_main_fit(xs)  # pins main_fit at prev max
	for (i,x) in enumerate(xs)
		xs[i] = max(x, xs[max(1, i-1)])
	end
	xs
end

# ╔═╡ 0b2d3f1c-3a78-4d7a-94dd-5eab9118faa2
minlen(xs) = xs |> ys -> [length(y) for y in ys] |> minimum

# ╔═╡ f91781de-3d67-48f5-b82a-90d91e105812
function matrixize(xs)
	ml = minlen(xs)
	xs = [x[1:ml] for x in xs]
	hcat(xs...)
end

# ╔═╡ 7d484f3b-57b6-4ea7-84d6-91be70916021
function revcumsum(vec)
	result = [first(vec)]
	for i in 2:length(vec) 
		push!(result, vec[i]-vec[i-1]) 
	end
	result
end

# ╔═╡ 8e03a713-5bb2-44c6-a61b-7fdbf7ea41a5
function readhiro(f, trials)
	dfs = []
	for i in trials
		stacked_df = DataFrame(CSV.File("$f$i/eval.csv"))
				
		df = unstack(stacked_df, :step, :metric, :value, allowduplicates=true)
		walltimes = stacked_df[1:nrow(df), :walltime]

		# train = unstack(DataFrame(CSV.File("$f$i/train.csv")), :step, :metric, :value, allowduplicates=true)

		# sps = mean(collect(skipmissing(train[!, "global_step/sec"])))

		# insertcols!(eval,       			  
		# 		    1,                		  
		# 		    :step_per_sec => [sps for _ in 1:nrow(eval)],
		# )

		walltimes = walltimes .- first(walltimes)
		insertcols!(df, 1, :time => revcumsum(walltimes))
		insertcols!(df, 1, :cumtime => walltimes)
		delete!(df, [nrow(df)])  # remove last row


		# eval[!, "step_per_sec"] .= sps

		df = dropmissing(df)
		
		push!(dfs, df)
	end
	dfs
end

# ╔═╡ 593931df-9fba-4548-ad2b-b48e8abec8ba
begin
	hiro_gather = readhiro("$pathprefix/gpu/hiro_orig/ant_gather/base_uvf/gather", 1:10)
	hiro_maze = readhiro("$pathprefix/gpu/hiro_orig/ant_maze/base_uvf/maze", 1:9)
	hiro_push = readhiro("$pathprefix/gpu/hiro_orig/ant_push_single/base_uvf/push", 1:10)

	"HIRO"
end

# ╔═╡ 08ee1e2d-0f88-4692-b9d8-62301cb06aa9
function clean(stacked_df, fix_gentime=false, missing_gentime=false)
	# removing preceding forward slash
	stacked_df.metric = map(x-> x[2:end], stacked_df.metric) 
	
	df = unstack(stacked_df, :step, :metric, :value) # rotating df
	
	if fix_gentime # fixing gen time logging bug - logged gentime in second half of df
		ndata = floor(Int32, (nrow(df) - 1) / 2)
		df[1:ndata, :gen_time_s] = df[ndata+1:ndata*2, :gen_time_s]  
		df = df[1:ndata, names(df)]  # cutting off at half the gens
	end

	if missing_gentime
		walltimes = stacked_df[1:nrow(df), :walltime]
		walltimes = walltimes .- first(walltimes)
		insertcols!(df, 4, :gen_time_s => revcumsum(walltimes))
	end
	
	insertcols!(df, 4, :gen_time => df.gen_time_s)
	df.gen_time_s = cumsum(df.gen_time_s) ./ 60 ./ 60
	
	df.main_fitness = fix_main_fit(df.main_fitness)
	df.total_steps = df.total_steps

	# changing column names
	rename!(df, Dict(
		"step" 					=> "Generation",
		"main_fitness" 			=> "Test Reward",
		"gen_time_s" 			=> "Time (h)",
		"total_steps" 			=> "Samples",
		"summarystat/1/mean" 	=> "Mean Controller Reward",
		"summarystat/2/mean" 	=> "Mean Primitive Reward"
	    )
	)

	delete!(df, [nrow(df)])
	dropmissing!(df)

	df
end

# ╔═╡ 519e5b4f-d204-4f3a-b473-52e45ec796f4
function readlog(f, trials; fix_gentime=false, missing_gentime=false)
	map(x->clean(x, fix_gentime, missing_gentime), [DataFrame(CSV.File("$f$i.csv")) for i in trials])
end

# ╔═╡ ec291a04-df1a-4b4a-89e1-6e15d5c6b8d2
begin
	gthr_base_dfs = readlog(gthr_tune_base_pth, 0:9)
	
	gthr_cd2_dfs = readlog(gthr_tune_cd2_pth, 0:2)
	gthr_cd8_dfs = readlog(gthr_tune_cd8_pth, 0:9)

	gthr_ep3_dfs = readlog(gthr_tune_ep3_pth, 0:2)
	gthr_ep10_dfs = readlog(gthr_tune_ep10_pth, 0:2)

	gthr_int10_dfs = readlog(gthr_tune_int10_pth, 0:2)
	gthr_int100_dfs = readlog(gthr_tune_int100_pth, 0:2)
	
	gthr_lr01_dfs = readlog(gthr_tune_lr01_pth, 0:2)
	gthr_lr0001_dfs = readlog(gthr_tune_lr0001_pth, 0:2)

	gthr_ppg512_dfs = readlog(gthr_tune_ppg512_pth, 0:2)
	gthr_ppg1000_dfs = readlog(gthr_tune_ppg1000_pth, 0:2)  # todo 1 is broken

	gthr_sigma02_dfs = readlog(gthr_tune_sigma02_pth, 0:2)
	gthr_sigma0002_dfs = readlog(gthr_tune_sigma0002_pth, 0:2)

	gather_tune = TuneRun(
		gthr_base_dfs, 
		gthr_cd2_dfs,
		gthr_cd8_dfs,
		gthr_ep3_dfs,
		gthr_ep10_dfs,
		gthr_int10_dfs,
		gthr_int100_dfs,
		gthr_lr01_dfs,
		gthr_lr0001_dfs,
		gthr_ppg512_dfs,
		gthr_ppg1000_dfs,
		gthr_sigma02_dfs,
		gthr_sigma0002_dfs
	)
	"Gather tune"
end

# ╔═╡ 9f9e14fe-8568-48c2-9106-4e265bc284d8
begin
	mze_base_dfs = readlog(mze_tune_base_pth, 0:9)
	mze_cd2_dfs = readlog(mze_tune_cd2_pth, 0:9)
	mze_cd8_dfs = readlog(mze_tune_cd8_pth, 0:9)
	mze_ep3_dfs = readlog(mze_tune_ep3_pth, 0:9)
	mze_ep10_dfs = readlog(mze_tune_ep10_pth, 0:9)
	mze_ep10_dfs=nothing
	
	mze_int10_dfs = readlog(mze_tune_int10_pth, 0:9)
	mze_int100_dfs = readlog(mze_tune_int100_pth, 0:9)
	# mze_lr01_dfs = readlog(mze_tune_lr01_pth, 0:8)
	# mze_lr0001_dfs = readlog(mze_tune_lr0001_pth, 0:9)
	mze_lr01_dfs=nothing
	mze_lr0001_dfs=nothing
	
	mze_ppg512_dfs = readlog(mze_tune_ppg512_pth, 0:9)
	mze_ppg1000_dfs = readlog(mze_tune_ppg1000_pth, 0:9)

	# mze_sigma02_dfs = readlog(mze_tune_sigma02_pth, 0:9)
	# mze_sigma0002_dfs = readlog(mze_tune_sigma0002_pth, 0:9)
	mze_sigma02_dfs=nothing
	mze_sigma0002_dfs=nothing
	
	maze_tune = TuneRun(
		mze_base_dfs, 
		mze_cd2_dfs,
		mze_cd8_dfs,
		mze_ep3_dfs,
		mze_ep10_dfs,
		mze_int10_dfs,
		mze_int100_dfs,
		mze_lr01_dfs,
		mze_lr0001_dfs,
		mze_ppg512_dfs,
		mze_ppg1000_dfs,
		mze_sigma02_dfs,
		mze_sigma0002_dfs
	)
	"Maze tune"
end

# ╔═╡ 5aa5ddd3-e896-47e7-bdd1-effda634168e
begin
	psh_base_dfs = readlog(psh_tune_base_pth, 0:6)
	
	psh_cd2_dfs = readlog(psh_tune_cd2_pth, 0:9)
	psh_cd8_dfs = readlog(psh_tune_cd8_pth, 0:9)

	psh_ep3_dfs = readlog(psh_tune_ep3_pth, 0:2)
	psh_ep10_dfs = readlog(psh_tune_ep10_pth, 0:2)

	psh_int10_dfs = readlog(psh_tune_int10_pth, 0:2)
	psh_int100_dfs = readlog(psh_tune_int100_pth, 0:2)
	
	psh_lr01_dfs = readlog(psh_tune_lr01_pth, 0:2)
	psh_lr0001_dfs = readlog(psh_tune_lr0001_pth, 0:2)

	psh_ppg512_dfs = readlog(psh_tune_ppg512_pth, 0:2)
	psh_ppg1000_dfs = readlog(psh_tune_ppg1000_pth, 0:2)

	psh_sigma02_dfs = readlog(psh_tune_sigma02_pth, 0:2)
	psh_sigma0002_dfs = readlog(psh_tune_sigma0002_pth, 0:2)

	push_tune = TuneRun(
		psh_base_dfs, 
		psh_cd2_dfs,
		psh_cd8_dfs,
		psh_ep3_dfs,
		psh_ep10_dfs,
		psh_int10_dfs,
		psh_int100_dfs,
		psh_lr01_dfs,
		psh_lr0001_dfs,
		psh_ppg512_dfs,
		psh_ppg1000_dfs,
		psh_sigma02_dfs,
		psh_sigma0002_dfs
	)
	"Push tune"
end

# ╔═╡ bed86641-3ef4-48f9-b27b-b7a98da0d2eb
begin
	# GATHER
	gather_nohot_dfs = readlog(gather_nohot_path, 0:8)
	gather_onehot_dfs = readlog(gather_onehot_path, 0:9)
	
	gather_pt_dfs = readlog(gather_pt_path, 0:9)
	# gather_ns_dfs = readlog(gather_ns_path, 0:4, missing_gentime=true)

	gather_10_dfs = readlog(gather_10_path, 0:4)
	gather_5_dfs = readlog(gather_5_path, 1:4)
	gather_single_df = readlog(gather_single_path, 0:0)
	all_nohot_dfs = vcat(gather_nohot_dfs, gather_10_dfs, gather_5_dfs)
	
	# MAZE
	maze_dfs = readlog(maze_path, 0:9)
	maze_pt_dfs = readlog(maze_pt_path, 5:9)
	# maze_ns_dfs = readlog(maze_ns_path, 0:4, missing_gentime=true)
	maze_5node_dfs = readlog(maze_5_node_path, 2:4)
	maze_10node_dfs = readlog(maze_10_node_path, 0:4)
	maze_single_df = readlog(maze_single_path, 0:0)
	all_maze_dfs = vcat(maze_dfs, maze_10node_dfs)

	# PUSH
	push_dfs = readlog(push_path, 0:9)
	push_pt_dfs = readlog(push_pt_path, 0:9)
	# push_ns_dfs = readlog(push_ns_path, 0:4)
	push_5node_dfs = readlog(push_5_node_path, 0:4)
	push_10node_dfs = readlog(push_10_node_path, 5:9)
	push_single_df = readlog(push_single_path, 0:0)
	all_push_dfs = vcat(push_dfs, push_10node_dfs)


	"SHES"
end

# ╔═╡ 22594eef-9489-4d68-9090-f4958e219b4f
function smooth(xs, ys, factor=0.5)
	model = loess(xs, ys, span=factor)
	vs = predict(model, xs)

	xs, vs
end

# ╔═╡ f878eb74-dc63-4a02-8aaf-2b5641c37c59
function mean_interp(dfs, x_ax, y_ax; inv=false)
	raw_ys = map(df->df[!, y_ax], dfs)
	raw_xs = map(df->df[!, x_ax], dfs)
	
	itps = [LinearInterpolation(raw_x, raw_y, extrapolation_bc=Line()) for (raw_x, raw_y) in zip(raw_xs, raw_ys)]

	mn = 0
	mx = mean([x[end] for x in raw_xs])
	len = raw_xs |> xs -> map(length, xs) |> mean |> x-> trunc(Int, x)
	
	xs = range(mn, mx, length=len)
	ys = matrixize([[itp(x) for x in xs] for itp in itps])
	mean_y = vec(mean(ys, dims=2))

	if inv
		LinearInterpolation(mean_y, xs)
	else
		LinearInterpolation(xs, mean_y)
	end
end

# ╔═╡ fc226d0f-5c1b-4eb8-a725-4b45d7151c97
function prep_data(raw_xs, raw_ys, smoothness)
	itps = [LinearInterpolation(raw_x, raw_y, extrapolation_bc=Line()) for (raw_x, raw_y) in zip(raw_xs, raw_ys)]

	mn = 0
	mx = mean([x[end] for x in raw_xs])
	len = raw_xs |> xs -> map(length, xs) |> minimum |> x-> trunc(Int, x)

	xs = range(mn, mx, length=len)
	ys = matrixize([[itp(x) for x in xs] for itp in itps])

	stddev = vec(std(ys, dims=2))
	mean_y = vec(mean(ys, dims=2))

	_, smooth_y = smooth(xs, mean_y, smoothness)
	_, smooth_stdev = smooth(xs, stddev, smoothness)
	
	xs, smooth_y, smooth_stdev
end

# ╔═╡ b9f2c016-93e1-440d-88c3-3f30921de1bd
function prep_data(dfs, x_ax, y_ax, smoothness; cutofftime=nothing)
	cutoffs = if cutofftime !== nothing
		# [1:findfirst(x -> x > cutofftime, df[!, "Time (h)"]) for df in dfs]

		x = []
		for df in dfs
			ftime = findfirst(x -> x > cutofftime, df[!, "Time (h)"])
			ftime = ftime === nothing ? nrow(df) : ftime
			push!(x, 1:ftime)
		end
		x
	else
		[1:nrow(df) for df in dfs]
	end
	
	raw_xs = map((df, cutoff)->df[!, x_ax][cutoff], dfs, cutoffs)
	raw_ys = map((df, cutoff)->df[!, y_ax][cutoff], dfs, cutoffs)
	
	prep_data(raw_xs, raw_ys, smoothness)
end

# ╔═╡ 468a639c-8a6c-417e-bac6-29c770a640f1
function plotexp(x,y,stdev, x_ax, y_ax, title, label, legendpos=:bottomright)
	plot(x, 
		y, 
		ribbon=stdev, 
		# fillalpha=.5, 
		title=title, 
		label=label, 
		legend=legendpos,
		dpi=500
	)
	xlabel!(x_ax)
	ylabel!(y_ax)
end

# ╔═╡ c4460ecb-31a8-439a-b7d7-3f08068926f0
function plotexp!(x, y, stdev, xtype, field, label)
	plot!(x, 
			y, 
			ribbon=stdev, 
			fillalpha=0.5, 
			label=label, 
		)
end

# ╔═╡ fa3f66c2-b52a-48fd-9e49-42e8815880fa
function plotexp(dfs, 
				 x_ax, 
				 y_ax, 
				 title, 
				 label, 
				 legendpos=:bottomright; 
				 cutofftime=nothing
				)
	
	x,y,stdev = prep_data(dfs, x_ax, y_ax, 0.2; cutofftime=cutofftime)
	plotexp(x,y,stdev,x_ax, y_ax, title, label, legendpos)
end

# ╔═╡ 6832f9ac-5105-47fb-a817-c1e5ce5dc6b8
function plotexp!(dfs, xtype, field, label; cutofftime=nothing)
	x,y,stdev = prep_data(dfs, xtype, field, 0.2; cutofftime=cutofftime)
	plotexp!(x, y, stdev, xtype, field, label)
end

# ╔═╡ c8b965da-3892-4040-89b3-ea8fdd23a5c6
plothiro!(pltable, _x, _y, _l; cutofftime=ct) = plothiro!(pltable; cutofftime=ct)

# ╔═╡ 258b40d2-4249-4087-9c75-503ca37ae9b3
struct Plotable
	df
	label
end

# ╔═╡ 2a3a5e40-c751-42be-b159-dd9c362f0d8e
function plotexp!(pable::Plotable, xtype, field, label; cutofftime=nothing)
	plotexp!(pable.df, xtype, field, label; cutofftime=cutofftime)
end

# ╔═╡ 456f5fd5-ef9c-4691-9c6d-b8ec408d894e
function plot_tune(exps, title_pref, save_pref, cut=ct)
	# PPG -----------------------------------------------------------
	plotexp(exps.base, "Time (h)", "Test Reward", "$title_pref Policies Per Generation", "256"; cutofftime=cut)
	plotexp!(exps.ppg512, "Time (h)", "Test Reward", "512"; cutofftime=cut)
	plotexp!(exps.ppg1000, "Time (h)", "Test Reward", "1000"; cutofftime=cut)

	savefig("$save_pref/tune/tune_ppg")
	
	# EPS -----------------------------------------------------------
	plotexp(exps.base, "Time (h)", "Test Reward", "$title_pref Episodes Per Policy", "5"; cutofftime=cut)
	plotexp!(exps.ep3, "Time (h)", "Test Reward", "3")  # 3 eps ends early
	# plotexp!(exps.ep10, "Time (h)", "Test Reward", "10"; cutofftime=cut)

	savefig("$save_pref/tune/tune_eps")
	
	# cdist ---------------------------------------------------------
	plotexp(exps.base, "Time (h)", "Test Reward", "$title_pref Target Distance", "4"; cutofftime=cut)
	plotexp!(exps.cd2, "Time (h)", "Test Reward", "2"; cutofftime=cut)
	plotexp!(exps.cd8, "Time (h)", "Test Reward", "8"; cutofftime=cut)

	savefig("$save_pref/tune/tune_cdist")
	
	# int -----------------------------------------------------------
	plotexp(exps.base, "Time (h)", "Test Reward", "$title_pref Controller Interval", "25"; cutofftime=cut)
	plotexp!(exps.int10, "Time (h)", "Test Reward", "10"; cutofftime=cut)
	# plotexp!(exps.int100, "Time (h)", "Test Reward", "100"; cutofftime=cut)

	savefig("$save_pref/tune/tune_int")
	
	# lr ------------------------------------------------------------
	# plotexp(exps.base, "Time (h)", "Test Reward", "$title_pref Learning Rate", "0.01"; cutofftime=cut)
	# plotexp!(exps.lr01, "Time (h)", "Test Reward", "0.1"; cutofftime=cut)
	# plotexp!(exps.lr0001, "Time (h)", "Test Reward", "0.001"; cutofftime=cut)

	# savefig("$save_pref/tune/tune_lr")

	# # σ -------------------------------------------------------------

	# plotexp(exps.base, "Time (h)", "Test Reward", "$title_pref Noise Standard Deviation (σ)", "0.02"; cutofftime=cut)
	# plotexp!(exps.sigma02, "Time (h)", "Test Reward", "0.2"; cutofftime=cut)
	# plotexp!(exps.sigma0002, "Time (h)", "Test Reward", "0.002"; cutofftime=cut)

	savefig("$save_pref/tune/tune_sigma")
end

# ╔═╡ b297c49a-62ea-440a-9aac-ab5dc1ec4639
begin 
	abstract type HiroPlotable end
	struct HiroGPlotable <: HiroPlotable 
		df
	end

	struct HiroMPlotable <: HiroPlotable 
		df
	end
	
	struct HiroPPlotable <: HiroPlotable 
		df
	end
end

# ╔═╡ 1013578e-e04c-4940-9d99-f05412e61206
function plothiro!(pltable::HiroGPlotable; cutofftime=ct)
	dfs = pltable.df

	cuts = [1:findfirst(x -> x / 3600 > cutofftime, df.cumtime) for df in dfs]
	xs = [df.cumtime[cut] / 3600 for (df, cut) in zip(dfs, cuts)]
	ys = [fix_main_fit(df[!, "Reward/average_eval_hrl_meta_reward"])[cut] 
	for (df, cut) in zip(dfs, cuts)]
	smooth_xs, smooth_ys, smooth_stdev = prep_data(xs, ys, .2)
	
	plot!(smooth_xs, smooth_ys, ribbon=smooth_stdev, fillalpha=.5, label="HIRO (ours)")
end

# ╔═╡ 0b437c56-6d7f-466e-9a25-4f596ee3f31c
function plothiro!(pltable::HiroMPlotable; cutofftime=ct)
	dfs = pltable.df

	cuts = [1:findfirst(x -> x / 3600 > cutofftime, df.cumtime) for df in dfs]
	xs = vcat(
		[df.cumtime[cut] / 3600 for (df, cut) in zip(dfs, cuts)],
		[df.cumtime[cut] / 3600 for (df, cut) in zip(dfs, cuts)],
		[df.cumtime[cut] / 3600 for (df, cut) in zip(dfs, cuts)]
	)
	ys = vcat(
		[fix_main_fit(df[!, "Reward/average_eval1_hrl_success"][cut]) 
			for (df, cut) in zip(dfs, cuts)],
		[fix_main_fit(df[!, "Reward/average_eval2_hrl_success"][cut]) 
			for (df, cut) in zip(dfs, cuts)],
		[fix_main_fit(df[!, "Reward/average_eval3_hrl_success"][cut]) 
			for (df, cut) in zip(dfs, cuts)],
	)

	smooth_xs, smooth_ys, smooth_stdev = prep_data(xs, ys, .2)
	
	plot!(smooth_xs, smooth_ys, ribbon=smooth_stdev, fillalpha=.5, label="HIRO (ours)")
end

# ╔═╡ 015d8f93-a60a-4d5e-a656-0c5e4a58e852
function plothiro!(pltable::HiroPPlotable; cutofftime=ct)
	dfs = pltable.df
	
	cuts = [1:findfirst(x -> x / 3600 > cutofftime, df.cumtime) for df in dfs]
	xs = [df.cumtime[cut] / 3600 for (df, cut) in zip(dfs, cuts)]
	ys = [fix_main_fit(df[!, "Reward/average_eval2_hrl_success"][cut]) for (df, cut) in zip(dfs, cuts)]

	smooth_xs, smooth_ys, smooth_stdev = prep_data(xs, ys, .2)
	|
	plot!(smooth_xs, smooth_ys, ribbon=smooth_stdev, fillalpha=.5, label="HIRO (ours)")
end

# ╔═╡ 965d3c00-bdc8-44cb-92bc-9a768cc8dead
label(p::Plotable) = p.label

# ╔═╡ ec02503b-8025-49a4-bf48-a2a532f17316
label(p) = nothing

# ╔═╡ 4e3e9c9e-3a4a-4716-a1ac-c116d43d1415
function multiplot(plottables, x_ax, y_ax, title)
	 p = plotexp(
		first(plottables).df, 
		x_ax,
		y_ax, 
		title, 
		first(plottables).label; 
		cutofftime=ct
	)
	
	for plottable in plottables[2:end]
		plt! = plottable isa HiroPlotable ? plothiro! : plotexp!
		plt!(
			plottable, 
			x_ax,
			y_ax, 
			label(plottable); 
			cutofftime=ct
		)
	end

	return p
end

# ╔═╡ dbffbce0-c1a9-4a02-8c5c-401d4740c77a
struct PlotableGroup  # TODO one node?
	one
	two
	five
	ten
	pt
	ns
	hiro
end

# ╔═╡ d0bb02aa-ae78-47b8-a939-d78f42d5cb92
begin
	g_plot = Plotable(gather_nohot_dfs, "SHES 48 cores")
	g_onehot_plot = Plotable(gather_onehot_dfs, "SHES one-hot 48 cores")
	g_5_plot = Plotable(gather_5_dfs, "SHES 120 cores")
	g_10_plot = Plotable(gather_10_dfs, "SHES 240 cores")
	g_pt_plot = Plotable(gather_pt_dfs, "SHES-TL")
	g_hiro_plot = HiroGPlotable(hiro_gather)
	g_pables = PlotableGroup(nothing, g_plot, g_5_plot, g_10_plot, g_pt_plot, nothing, g_hiro_plot)
	
	m_plot = Plotable(maze_dfs, "SHES 48 cores")
	m_5_plot = Plotable(maze_5node_dfs, "SHES 120 cores")
	m_10_plot = Plotable(maze_10node_dfs, "SHES 240 cores")
	m_pt_plot = Plotable(maze_pt_dfs, "SHES-TL")
	m_hiro_plot = HiroMPlotable(hiro_maze)
	m_pables = PlotableGroup(nothing, m_plot, m_5_plot, m_10_plot, m_pt_plot, nothing, m_hiro_plot)

	p_plot = Plotable(push_dfs, "SHES 48 cores")
	p_5_plot = Plotable(push_5node_dfs, "SHES 120 cores")
	p_10_plot = Plotable(push_10node_dfs, "SHES 240 cores")
	p_pt_plot = Plotable(push_pt_dfs, "SHES-TL")
	p_hiro_plot = HiroPPlotable(hiro_push)
	p_pables = PlotableGroup(nothing, p_plot, p_5_plot, p_10_plot, p_pt_plot, nothing, p_hiro_plot)

	"Plotables"
end

# ╔═╡ d18d3765-3c63-4ce7-9a69-3b820442f161
plotmain(pable; yax=TR) = multiplot([pable.two, pable.ten, pable.hiro], TIME, yax, "SHES vs gradient based methods")

# ╔═╡ 15f7aceb-8388-4d2d-9a8b-0f4a87f2f656
plotnodes(pable; yax=TR) = multiplot([pable.two, pable.five, pable.ten], TIME, yax, "SHES Multicore comparison")

# ╔═╡ 507a87b4-47ec-4cdf-80ca-6f7e87afb876
plotexts(pable; yax=TR) = multiplot([pable.two, pable.pt], TIME, yax, "SHES vs Extensions")

# ╔═╡ 8090773f-e2c3-4c44-8428-d7284221050d
plotnodes(m_pables;yax=CR)

# ╔═╡ b6dd438c-4e57-4b52-a1f5-98d29f16acc6
md"# Ant gather"

# ╔═╡ f9551d45-b4c1-43a4-a76e-afd7ff53bd95


# ╔═╡ fafa619c-8187-4ffe-a6ec-994957dfb7d4
md"### Tune"

# ╔═╡ 97293ad9-4ef1-46ec-ba4c-ea593a66d449
plot_tune(gather_tune, "Ant Gather:", "imgs/gather")

# ╔═╡ b60af232-58db-488b-b48e-baf6cd412d15
md"### Test reward"

# ╔═╡ ecb988c8-844b-4a69-9870-613398238f65
begin
	t1 = "Ant Gather: Test Reward Per Environment Steps"

	# all_nohot_dfs = vcat(gather_nohot_dfs, gather_5_dfs, gather_10_dfs)
	
	plotexp(gather_nohot_dfs, "Samples", "Test Reward", t1, "SHES"; cutofftime=ct)
	plotexp!(gather_onehot_dfs, "Samples", "Test Reward", "SHES one-hot"; cutofftime=ct)

	gather_other_xs = [0, 1.68e9]
	plot!(gather_other_xs, [3.02, 3.02], label="HIRO (theirs)")
	plot!(gather_other_xs, [0.85, 0.85], label="FuN")
	plot!(gather_other_xs, [1.92, 1.92], label="SNN4HRL")
	plot!(gather_other_xs, [1.42, 1.42], label="VIME")

	# savefig("imgs/gather/gather-tr-steps")
end

# ╔═╡ 0ba44efd-637e-4d23-8236-ffd3207f549c
multiplot([g_plot, g_onehot_plot], TIME, TR, "Ant Gather: SHES vs SHES-onehot")

# ╔═╡ 0fa2bbcd-0b9e-4f79-8679-0e7d698d0068
begin
	t5 = "Ant Gather: Test Score"
	
	plotexp(gather_nohot_dfs, "Time (h)", "Test Reward", t5, "SHES 48 cores", :bottomright; cutofftime=ct)

	# plotexp!(gather_onehot_dfs, "Time (h)", "Test Reward", "SHES one-hot"; cutofftime=ct)
	# plotexp!(gather_5_dfs, "Time (h)", "Test Reward", "SHES 120 cores";)
	plotexp!(gather_10_dfs, "Time (h)", "Test Reward", "SHES 240 cores"; cutofftime=ct)

	# plotexp!(gather_pt_dfs, "Time (h)", "Test Reward", "Pretrained"; cutofftime=ct)


	plothiro_gather!(hiro_gather)
	
	gather_other_xs2 = [0, ct]
	plot!(gather_other_xs2, [3.02, 3.02], label="HIRO (theirs)")
	plot!(gather_other_xs2, [0.85, 0.85], label="FuN")
	# plot!(gather_other_xs2, [1.92, 1.92], label="SNN4HRL")
	# plot!(gather_other_xs2, [1.42, 1.42], label="VIME")

	savefig("imgs/gather/gather-tr-time")
end

# ╔═╡ efca9de3-e1f8-47de-8c7f-47f9f6b72ba9
md"### Pretraining+NS"

# ╔═╡ a49117d2-562f-4af2-be5e-9f56423fe9c0
begin
	t4 = "Ant Gather: Training Reward"
	
	plotexp(gather_nohot_dfs, "Time (h)", CR, t4, "SHES 48 cores"; cutofftime=ct)
	plotexp!(gather_pt_dfs, "Time (h)", CR, "SHES-TL 48 cores"; cutofftime=ct)
	# plotexp!(gather_ns_dfs, "Time (h)", "Test Reward", "Novelty"; cutofftime=ct)
	
	savefig("imgs/gather/gather-tl")
end

# ╔═╡ 9c6878e0-741d-4030-9d4a-94fa40184221
md"### Train reward"

# ╔═╡ ca002d4f-666b-4126-a94f-b664113086d3
begin
	t3 = "Ant Gather: Mean Primitive Training Reward"
	
	plotexp(gather_nohot_dfs, "Time (h)", "Mean Primitive Reward", t3, "SHES"; cutofftime=ct)
	plotexp!(gather_onehot_dfs, "Time (h)", "Mean Primitive Reward", "SHES one-hot"; cutofftime=ct)
	plotexp!(gather_ns_dfs, "Time (h)", "Mean Primitive Reward", "NS"; cutofftime=ct)
	plotexp!(gather_pretrained_dfs, "Time (h)", "Mean Primitive Reward", "Pretrain"; cutofftime=ct)

	savefig("imgs/gather/gather-prim")
end

# ╔═╡ 379ac95f-6d0a-446a-9dec-24135d79ea2c
md"### Speed up"

# ╔═╡ e4cc1434-ef87-43f0-80c9-e84c9690dda5
mean_gt(dfs) = mean(map(df -> mean(df.gen_time), dfs))

# ╔═╡ 0bfb4725-966c-48a2-9efe-253355030ae8
md"# Ant maze"

# ╔═╡ 0e47ea2c-0041-4209-89af-e6070ff2727f
md"### Test reward"

# ╔═╡ ae589eef-898c-4c6f-957d-ccb240e13c55
begin
	t2 = "Ant Maze: Test Reward Per Environmental Steps"

	plotexp(all_maze_dfs, "Samples", "Test Reward", t2, "SHES", :topright; cutofftime=ct)

	maze_other_xs = [0, 8e9] 
	plot!(maze_other_xs, [0.99,0.99], label="HIRO (theirs)")
	plot!(maze_other_xs, [0.16, 0.16], label="FuN")
	plot!(maze_other_xs, [0, 0], label="SNN4HRL+VIME")

	savefig("imgs/maze/maze-tr-steps")
end

# ╔═╡ b48b0893-266b-4937-af92-16045dcec171
begin
	t7 = "Ant Maze: Test Score"
	
	plotexp(maze_dfs, "Time (h)", "Test Reward", t7, "SHES 48 cores", :right; cutofftime=ct)
	# plotexp!(maze_5node_dfs, "Time (h)", "Test Reward", "120"; cutofftime=ct)
	plotexp!(maze_10node_dfs, "Time (h)", "Test Reward", "SHES 240 cores")

	plothiro_maze!(hiro_maze[1:7])

	maze_other_xs2 = [0, ct]
	plot!(maze_other_xs2, [0.99, 0.99], label="HIRO")

	plot!(maze_other_xs2, [0.16, 0.16], label="FuN")
	# plot!(maze_other_xs2, [0, 0], label="SNN4HRL+VIME")

	savefig("imgs/maze/maze-tr-time")
end

# ╔═╡ 67939190-1a19-45f9-a27f-01021b497c07
md"### Train Reward"

# ╔═╡ 8de599b3-39a1-4e7c-af38-f1e40864d059
begin
	t8 = "Ant Maze: Training Reward"
	
	plotexp(maze_dfs, "Time (h)", "Mean Controller Reward", t8, "SHES 48 cores"; cutofftime=ct)
	plotexp!(maze_pt_dfs, "Time (h)", "Mean Controller Reward", "SHES-TL 48 cores"; cutofftime=7.5)
	# plotexp!(maze_ns_dfs, "Time (h)", "Mean Primitive Reward", "NS"; cutofftime=ct)
	
	savefig("imgs/maze/maze-pt-train")
end

# ╔═╡ 4c4c7482-867b-474c-abea-8dc9ea82c890
md"### Pretrain + NS"

# ╔═╡ 6158877d-4494-44d0-836d-81829ae391bf
begin
	t9 = "Ant Maze: Pretrained vs Novelty vs Base SHES"
	
	plotexp(maze_dfs, "Time (h)", "Test Reward", t9, "SHES"; cutofftime=ct)
	plotexp!(maze_pt_dfs, "Time (h)", "Test Reward", "Pretrained")
	# plotexp!(maze_ns_dfs, "Time (h)", "Test Reward", "NS"; cutofftime=ct)
	
	# savefig("imgs/maze/maze-pt-test")
end

# ╔═╡ 09bfe3f2-3a07-4701-a169-76d6b12490ba
md"### Speedup"

# ╔═╡ b3c916f1-fed1-4ce7-a0cc-84b16877dcb3
md"### Tune"

# ╔═╡ e23d4c2f-620f-4fc4-b56a-0a285639e4ea
plot_tune(maze_tune, "Ant Maze:", "imgs/maze")

# ╔═╡ be0d4809-3b6a-4985-bb2c-117c14bd31bb
map(df -> maximum(df[!, "Time (h)"]), maze_tune.ppg512)

# ╔═╡ a16e058e-728e-4a24-a21a-aaafb5a6c183
md"# Ant Push"

# ╔═╡ a07694f7-bfcb-4a0e-87a4-7d5fddeeb5c8
md"### Test Reward"

# ╔═╡ d8669c2a-e38b-44f7-aeec-7920dc5abf1c
begin
	t11 = "Ant Push: Test Reward Per Environmental Steps"

	plotexp(all_push_dfs, "Samples", "Test Reward", t11, "SHES (any processors)"; cutofftime=ct)
	
	push_other_xs = [0, 7.63e9]
	plot!(push_other_xs, [0.92,0.92], label="HIRO")
	plot!(push_other_xs, [0.56, 0.56], label="FuN")
	plot!(push_other_xs, [0.02, 0.02], label="SNN4HRL+VIME")

	savefig("imgs/push/push-tr-steps")
end

# ╔═╡ 947dcf8c-3799-47d7-9d78-1058e111ea60
begin
	t12 = "Ant Push: Test Score"

	plotexp(push_dfs, "Time (h)", "Test Reward", t12, "SHES 48 cores"; cutofftime=ct)
	# plotexp!(push_5node_dfs, "Time (h)", "Test Reward", "SHES 120 cores"; cutofftime=ct)
	plotexp!(push_10node_dfs, "Time (h)", "Test Reward", "SHES 240 cores"; cutofftime=ct)
	# plotexp!(push_pt_dfs, "Time (h)", "Test Reward", "SHES-TL 48 cores")

	plothiro_push!(hiro_push; cutofftime=9.92)
	push_other_xs_t = [0, ct]
	plot!(push_other_xs_t, [0.92,0.92], label="HIRO")
	plot!(push_other_xs_t, [0.16, 0.16], label="FuN")
	# plot!(push_other_xs_t, [0, 0], label="SNN4HRL+VIME")

	savefig("imgs/push/push-tr-time")
end

# ╔═╡ eea5569f-cea7-493b-965d-91bab81dee7c
md"### Pretrain + NS"

# ╔═╡ 48f0f7c3-1492-4d94-a755-58ede5b3d3fd
begin
	t13 = "Ant Push: Training Reward"
	
	plotexp(push_dfs, "Time (h)", CR, t13, "SHES 48 cores")
	plotexp!(push_pt_dfs, "Time (h)", CR, "SHES-TL 48 cores")
	# plotexp!(push_ns_dfs, "Samples", "Test Reward", "NS")
	
	savefig("imgs/push/push-tl")
end

# ╔═╡ 07861635-0ce0-4315-9259-91956e6f5f1a
md"### Prim reward"

# ╔═╡ b2419d55-5d0e-43a5-a1d5-f10e645ae388
begin
	t14 = "Ant Push: Mean Primitive Training Reward"
	
	plotexp(push_dfs, "Time (h)", "Mean Primitive Reward", t14, "SHES")
	plotexp!(push_pt_dfs, "Time (h)", "Mean Primitive Reward", "SHES")
	plotexp!(push_ns_dfs, "Time (h)", "Mean Primitive Reward", "SHES")
	
	savefig("imgs/push/push-prim")
end

# ╔═╡ bccedaad-c533-4dbb-90f8-79140e5af04b
md"### Speedup"

# ╔═╡ 50f946d3-937e-448d-835e-e17e676e67c4
md"### Tune"

# ╔═╡ 868412e5-f725-415b-a7ab-d6bed0e15b13
plot_tune(push_tune, "Ant Push:", "imgs/push", 1)

# ╔═╡ fbb132aa-4f45-4f22-9339-25511aaa840b
md"##### Scratch"

# ╔═╡ 8d5baceb-f1e3-451a-92d5-cfabebaea4e7
mean_interp(push_10node_dfs, "Time (h)", "Test Reward", inv=true)(0.92)

# ╔═╡ a1cd331f-2d70-4802-8d76-7038268da1e7
begin
	raw_xs = map(df->df[!, "Samples"], gather_onehot_dfs)
	mean([x[end] for x in raw_xs])
end

# ╔═╡ b4edcf02-66f9-4ff7-a168-ea3100e5a5c6
function inc_run_names(path, inc)
	for f in reverse(readdir(path))
		cn = parse(Int, split(f, "_")[end])
		mv(joinpath(path, f), joinpath(path, "$(f[1:end-1])$(cn+inc)"))
	end
end

# ╔═╡ 28449291-d39e-4fae-bb16-7faacd8f9da0
# inc_run_names("../ScalableHrlEs/tensorboard_logs/remote/maze/tune/new/", 3)

# ╔═╡ cfdeb877-0ffa-4bc6-8d28-0478734f593e
begin
	g_48_mgt = mean_gt(gather_nohot_dfs)
	g_1_mgt = mean_gt(gather_single_df)
	g_240_mgt = mean_gt(gather_10_dfs)

	# plot([1,240], [1,240], title="Speedup vs core count", label="Linear speedup", yaxis=:log, xaxis=:log, legend=:topleft)
	plot([1,240], [1,240], title="Speedup vs core count", label="Linear speedup", legend=:topleft)
	plot!([1, 48, 240], g_1_mgt ./ [g_1_mgt, g_48_mgt, g_240_mgt], label="SHES speedup")
	xlabel!("Speedup")
	ylabel!("Cores")

	savefig("imgs/speedup")
end

# ╔═╡ 4aeea9c5-5ffd-465f-8a11-65d2feba1286
begin
	m_1_mgt = mean_gt(maze_single_df)
	m_24_mgt = mean_gt(maze_tune.ppg1000)
	m_48_mgt = mean_gt(maze_dfs)
	m_240_mgt = mean_gt(maze_10node_dfs)

	plot([1,240], [1,240])
	plot!([1, 24,  48, 240], m_1_mgt ./ [m_1_mgt, m_24_mgt, m_48_mgt, m_240_mgt] )
end

# ╔═╡ 04cd450c-d04d-41a8-9e3e-add8c407a9b8
begin
	p_48_mgt = mean_gt(push_dfs)
	p_1_mgt = mean_gt(push_single_df)
	p_240_mgt = mean_gt(push_10node_dfs)

	plot([1,240], [1,240])
	plot!([1, 48, 240], p_1_mgt ./ [p_1_mgt, p_48_mgt, p_240_mgt] )
end

# ╔═╡ 3470328c-77bc-4dfd-952c-51346bb743c5
begin
	# base
	gather_max = [maximum(df[!, "Test Reward"]) for df in all_nohot_dfs]
	# maze_max = [maximum(df[!, "Test Reward"]) for df in all_maze_dfs]
	push_max = [maximum(df[!, "Test Reward"]) for df in all_push_dfs]

	# exts
	gather_pt_max = [maximum(df[!, "Test Reward"]) for df in gather_pt_dfs]
	# gather_ns_max = [maximum(df[!, "Test Reward"]) for df in gather_ns_dfs]

	# maze_pt_max = [maximum(df[!, "Test Reward"]) for df in maze_pt_dfs]
	# maze_ns_max = [maximum(df[!, "Test Reward"]) for df in maze_ns_dfs]

	push_pt_max = [maximum(df[!, "Test Reward"]) for df in push_pt_dfs]
	# push_ns_max = [maximum(df[!, "Test Reward"]) for df in push_ns_dfs]

	# hiro
	hiro_gather_max = [maximum(df[!, "Reward/average_eval_hrl_meta_reward"]) for df in hiro_gather]
	hiro_push_max = [maximum(df[!, "Reward/average_eval2_hrl_success"]) for df in hiro_push]
	hiro_maze_max = [maximum(df[!, "Reward/average_eval$(i)_hrl_success"]) for df in hiro_maze for i in 1:3]
		
	MannWhitneyUTest(gather_max, gather_pt_max)  # reject = one of the two vals are bigger
end

# ╔═╡ 9e27690c-0557-468b-a754-ba051c0e581c
[maximum(df[!, "Test Reward"]) for df in push_10node_dfs] / 0.92

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HypothesisTests = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
Loess = "4345ca2d-374a-55d4-8d30-97f9976e7612"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CSV = "~0.9.11"
DataFrames = "~1.3.1"
HypothesisTests = "~0.10.6"
Interpolations = "~0.13.5"
Loess = "~0.5.4"
Plots = "~1.25.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "49f14b6c56a2da47608fe30aed711b5882264d7a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.11"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4c26b4e9e91ca528ea212927326ece5918a04b47"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.2"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "cfdfef912b7f93e4b848e80b9befdf9e331bc05a"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.1"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "6a8dc9f82e5ce28279b6e3e2cea9421154f5bd0d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.37"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f97acd98255568c3c9b416c5a3cf246c1315771b"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.63.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HypothesisTests]]
deps = ["Combinatorics", "Distributions", "LinearAlgebra", "Random", "Rmath", "Roots", "Statistics", "StatsBase"]
git-tree-sha1 = "dc9bb7abfa265e0cf030635315184a476a2dd5f3"
uuid = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
version = "0.10.6"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "8d70835a3759cdd75881426fced1508bb7b7e1b6"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b15fc0a95c564ca2e0a7ae12c1f095ca848ceb31"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.5"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "46efcea75c890e5d820e670516dc156689851722"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.4"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "f755f36b19a5116bb580de457cda0c140153f283"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.6"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "d7fa6237da8004be601e19bd6666083056649918"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "68604313ed59f0408313228ba09e79252e4b2da8"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.2"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "7eda8e2a61e35b7f553172ef3d9eaa5e4e76d92e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.3"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "8f82019e525f4d5c669692772a6f4b0a58b06a6a"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.2.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "ee885e0f773804f046fd43d0d4ace305b3d540e2"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "1.3.13"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "244586bc07462d22aed0113af9c731f2a518c93e"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.10"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "0afd9e6c623e379f593da01f20590bacc26d1d14"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═cd33cc66-1df7-11ec-102a-d5816d93431c
# ╠═b29d2430-3950-4544-8daf-57be549a690f
# ╟─1b4dc0db-9f47-4a94-9bbc-b70c30972997
# ╟─006035ef-08f6-4f9f-9dfb-043abfaf6af6
# ╟─c1dae931-ab39-4d98-bb9d-0453ce0e9e52
# ╟─a36250ad-7318-4e57-a6ae-c36e782d2214
# ╟─7df4d38a-4b25-437c-975c-78010c3729c1
# ╟─bdbd896d-0761-4d30-82ab-928889953d3f
# ╠═3163445d-ce5e-424a-961a-5ca03a34f177
# ╟─27ea31d5-52ab-4186-8859-48e912abbe07
# ╟─1341a522-35b1-4d12-acc5-5455cff02992
# ╟─a28bbacf-5fa1-4a60-9ac5-66ca96dd0765
# ╟─4bab0152-d53f-4bb9-a38a-49930797c632
# ╟─ec291a04-df1a-4b4a-89e1-6e15d5c6b8d2
# ╠═9f9e14fe-8568-48c2-9106-4e265bc284d8
# ╟─5aa5ddd3-e896-47e7-bdd1-effda634168e
# ╠═bed86641-3ef4-48f9-b27b-b7a98da0d2eb
# ╟─593931df-9fba-4548-ad2b-b48e8abec8ba
# ╟─35dc88a2-9b9a-4d57-8036-ca198020276c
# ╟─519e5b4f-d204-4f3a-b473-52e45ec796f4
# ╟─8e03a713-5bb2-44c6-a61b-7fdbf7ea41a5
# ╟─23463959-ea89-4276-80a1-92cb3ff105b0
# ╟─08ee1e2d-0f88-4692-b9d8-62301cb06aa9
# ╟─0b2d3f1c-3a78-4d7a-94dd-5eab9118faa2
# ╟─f91781de-3d67-48f5-b82a-90d91e105812
# ╟─7d484f3b-57b6-4ea7-84d6-91be70916021
# ╟─22594eef-9489-4d68-9090-f4958e219b4f
# ╟─f878eb74-dc63-4a02-8aaf-2b5641c37c59
# ╟─fc226d0f-5c1b-4eb8-a725-4b45d7151c97
# ╠═b9f2c016-93e1-440d-88c3-3f30921de1bd
# ╟─468a639c-8a6c-417e-bac6-29c770a640f1
# ╠═c4460ecb-31a8-439a-b7d7-3f08068926f0
# ╟─fa3f66c2-b52a-48fd-9e49-42e8815880fa
# ╠═6832f9ac-5105-47fb-a817-c1e5ce5dc6b8
# ╠═2a3a5e40-c751-42be-b159-dd9c362f0d8e
# ╟─456f5fd5-ef9c-4691-9c6d-b8ec408d894e
# ╠═c8b965da-3892-4040-89b3-ea8fdd23a5c6
# ╟─1013578e-e04c-4940-9d99-f05412e61206
# ╟─0b437c56-6d7f-466e-9a25-4f596ee3f31c
# ╟─015d8f93-a60a-4d5e-a656-0c5e4a58e852
# ╠═258b40d2-4249-4087-9c75-503ca37ae9b3
# ╟─b297c49a-62ea-440a-9aac-ab5dc1ec4639
# ╟─965d3c00-bdc8-44cb-92bc-9a768cc8dead
# ╟─ec02503b-8025-49a4-bf48-a2a532f17316
# ╠═4e3e9c9e-3a4a-4716-a1ac-c116d43d1415
# ╠═dbffbce0-c1a9-4a02-8c5c-401d4740c77a
# ╟─d0bb02aa-ae78-47b8-a939-d78f42d5cb92
# ╠═d18d3765-3c63-4ce7-9a69-3b820442f161
# ╠═15f7aceb-8388-4d2d-9a8b-0f4a87f2f656
# ╠═507a87b4-47ec-4cdf-80ca-6f7e87afb876
# ╠═8090773f-e2c3-4c44-8428-d7284221050d
# ╟─b6dd438c-4e57-4b52-a1f5-98d29f16acc6
# ╠═f9551d45-b4c1-43a4-a76e-afd7ff53bd95
# ╟─fafa619c-8187-4ffe-a6ec-994957dfb7d4
# ╠═97293ad9-4ef1-46ec-ba4c-ea593a66d449
# ╟─b60af232-58db-488b-b48e-baf6cd412d15
# ╠═ecb988c8-844b-4a69-9870-613398238f65
# ╠═0ba44efd-637e-4d23-8236-ffd3207f549c
# ╠═0fa2bbcd-0b9e-4f79-8679-0e7d698d0068
# ╟─efca9de3-e1f8-47de-8c7f-47f9f6b72ba9
# ╠═a49117d2-562f-4af2-be5e-9f56423fe9c0
# ╟─9c6878e0-741d-4030-9d4a-94fa40184221
# ╠═ca002d4f-666b-4126-a94f-b664113086d3
# ╟─379ac95f-6d0a-446a-9dec-24135d79ea2c
# ╠═e4cc1434-ef87-43f0-80c9-e84c9690dda5
# ╟─0bfb4725-966c-48a2-9efe-253355030ae8
# ╟─0e47ea2c-0041-4209-89af-e6070ff2727f
# ╠═ae589eef-898c-4c6f-957d-ccb240e13c55
# ╠═b48b0893-266b-4937-af92-16045dcec171
# ╟─67939190-1a19-45f9-a27f-01021b497c07
# ╠═8de599b3-39a1-4e7c-af38-f1e40864d059
# ╟─4c4c7482-867b-474c-abea-8dc9ea82c890
# ╠═6158877d-4494-44d0-836d-81829ae391bf
# ╟─09bfe3f2-3a07-4701-a169-76d6b12490ba
# ╟─b3c916f1-fed1-4ce7-a0cc-84b16877dcb3
# ╠═e23d4c2f-620f-4fc4-b56a-0a285639e4ea
# ╠═be0d4809-3b6a-4985-bb2c-117c14bd31bb
# ╟─a16e058e-728e-4a24-a21a-aaafb5a6c183
# ╟─a07694f7-bfcb-4a0e-87a4-7d5fddeeb5c8
# ╠═d8669c2a-e38b-44f7-aeec-7920dc5abf1c
# ╠═947dcf8c-3799-47d7-9d78-1058e111ea60
# ╟─eea5569f-cea7-493b-965d-91bab81dee7c
# ╠═48f0f7c3-1492-4d94-a755-58ede5b3d3fd
# ╟─07861635-0ce0-4315-9259-91956e6f5f1a
# ╠═b2419d55-5d0e-43a5-a1d5-f10e645ae388
# ╟─bccedaad-c533-4dbb-90f8-79140e5af04b
# ╟─50f946d3-937e-448d-835e-e17e676e67c4
# ╠═868412e5-f725-415b-a7ab-d6bed0e15b13
# ╟─fbb132aa-4f45-4f22-9339-25511aaa840b
# ╠═8d5baceb-f1e3-451a-92d5-cfabebaea4e7
# ╠═a1cd331f-2d70-4802-8d76-7038268da1e7
# ╠═b4edcf02-66f9-4ff7-a168-ea3100e5a5c6
# ╠═28449291-d39e-4fae-bb16-7faacd8f9da0
# ╠═cfdeb877-0ffa-4bc6-8d28-0478734f593e
# ╠═4aeea9c5-5ffd-465f-8a11-65d2feba1286
# ╠═04cd450c-d04d-41a8-9e3e-add8c407a9b8
# ╠═3470328c-77bc-4dfd-952c-51346bb743c5
# ╠═9e27690c-0557-468b-a754-ba051c0e581c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
