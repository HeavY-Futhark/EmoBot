<script lang="ts">
	import { Avatar, ListBox, ListBoxItem } from '@skeletonlabs/skeleton';
	import { onMount } from 'svelte';

	let paintings: { name: string; img: unknown }[] = [];
	let currentMessage = '';

	onMount(async () => {
		const paintingsImports = import.meta.glob(
			'$assets/paintings/*.{avif,gif,heif,jpeg,jpg,png,tiff,webp}',
			{
				query: {
					enhanced: true
				}
			}
		);
		console.log(paintingsImports);
		const imageNameRegex = /^.*\/(.+)?\..+$/;

		for (const [key, paintingImport] of Object.entries(paintingsImports)) {
			const painting = await paintingImport();
			const paintingName = key.match(imageNameRegex)[1];
			paintings = [...paintings, { img: painting.default, name: paintingName }];
		}
	});
	$: console.log(paintings);
	let selectedPaintings: [] = [];
</script>

<div class="container h-full mx-auto flex justify-center items-start m-8">
	<div class="space-y-10 w-full text-center flex flex-col items-center">
		<h2 class="h2">Ontologie</h2>
		<div class="flex gap-4">
			<button type="button" class="btn variant-filled-secondary">Remplir</button>
			<button type="button" class="btn variant-filled-secondary">Afficher</button>
			<button type="button" class="btn variant-filled-secondary">Vider</button>
		</div>
		<div class="flex w-full">
			<div class="space-y-10">
				<h2 class="h2">Visiter un tableau</h2>
				<ListBox rounded="rounded-xl" spacing="grid grid-cols-2 md:grid-cols-3 gap-4">
					{#each paintings as painting (painting.name)}
						<ListBoxItem bind:group={selectedPaintings} name={painting.name} value={painting.name}>
							<div class="w-64 h-44">
								<enhanced:img
									src={painting.img}
									alt={painting.name}
									class="mx-auto max-h-full max-w-full"
								/>
							</div>
						</ListBoxItem>
					{/each}
				</ListBox>
			</div>
			<div class="space-y-10 flex-grow">
				<h2 class="h2">Parler au guide</h2>

				<div class="grid grid-cols-[auto_1fr] gap-2">
					<Avatar src="https://i.pravatar.cc/" width="w-12" />
					<div class="card p-4 variant-soft rounded-tl-none space-y-2">
						<header class="flex justify-between items-center">
							<p class="font-bold">Avatar</p>
							<small class="opacity-50">Timestamp</small>
						</header>
						<p>Message</p>
					</div>
				</div>

				<div
					class="input-group input-group-divider grid-cols-[auto_1fr_auto] rounded-container-token"
				>
					<button class="input-group-shim">+</button>

					<textarea
						bind:value={currentMessage}
						class="bg-transparent border-0 ring-0"
						name="prompt"
						id="prompt"
						placeholder="Write a message..."
						rows="1"
					/>
					<button class="variant-filled-primary">Send</button>
				</div>
			</div>
		</div>
	</div>
</div>

<style lang="postcss">
</style>
