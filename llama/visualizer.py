import torch
from PIL import Image, ImageDraw


class AttentionVisualizer():
    VISUALIZER_ID = 0

    def __init__(self):
        self._tokens = []
        self._attention_scores = []
        self._id = self.get_visualizer_id()

    @classmethod
    def get_visualizer_id(cls):
        id = cls.VISUALIZER_ID
        cls.VISUALIZER_ID += 1
        return id

    def add_tokens(self, tokens):
        self._tokens.append(tokens)

    def add_scores(self, scores):
        self._attention_scores.append(scores)

        # TODO: sticking this here for now....
        self.generate_gif()

    def _generate_image(self, tokens, attention_scores):
        x_offset_init = 10
        x_offset = x_offset_init
        y_offset = 10

        # Grab what the final string's rendered width will be
        tmp_img = Image.new(mode="RGB", size=(0, 0))
        tmp_draw = ImageDraw.Draw(tmp_img)
        final_text_size = tmp_draw.textbbox((0, 0), " ".join(tokens))  # 0 offset because we only need the text width/height

        # Construct the real image
        img = Image.new(mode="RGB", size=(final_text_size[2] + 2 * x_offset, final_text_size[3] + 2 * y_offset), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Add tokens, giving them color proportional to the attention given
        for token_id, token in enumerate(tokens):
            score = attention_scores[token_id].item()
            token = token + " "
            token_render_size = draw.textbbox((0, 0), token)

            if token == "\n ":  # TODO: lol hacky
                x_offset = x_offset_init
                y_offset += token_render_size[3]  # TODO: check
            else:
                assert "\n" not in token
                x_offset += token_render_size[2]

            text_pos = (x_offset, y_offset)
            text_bbox = draw.textbbox(text_pos, token)

            draw.rectangle(text_bbox, fill=(255, int(255 * (1 - score)), int(255 * (1 - score))))
            draw.text(text_pos, token, fill=(0, 0, 0))

        #img.show()
        return img

    def _generate_images(self, tokens, attention_scores):
        assert len(tokens) == attention_scores.shape[-1]
        all_head_images = []

        for head_id in range(1):  # TODO just doing the first head for now. attention_scores.shape[0]):
            head_images = []

            for timestep_id in range(attention_scores.shape[1]):
                # "timestep" is sort of misleading, but it's intended to mean "the point when a token is processed"

                image = self._generate_image(tokens, attention_scores[head_id][timestep_id])
                head_images.append(image)

            all_head_images.append(head_images)
        return all_head_images

    def generate_gif(self):
        assert len(self._tokens) == len(self._attention_scores), "A set of tokens needs to be added per attention."
        cumulative_tokens = []
        batch_id = 0
        all_head_images = []

        for timestep, token_set in enumerate(self._tokens):
            attention_scores = self._attention_scores[timestep][batch_id]
            token_set = token_set[0]

            attention_count = attention_scores.shape[-1]
            cumulative_tokens.extend(token_set)

            if len(token_set) != attention_count:
                # In this case, assume we're caching
                token_set = cumulative_tokens #torch.concat(cumulative_tokens, dim=0)

            all_head_images_for_timestep = self._generate_images(token_set, attention_scores)

            for head_id, head_images in enumerate(all_head_images_for_timestep):
                if len(all_head_images) <= head_id:  # Just lazy because I don't feel like pre-allocating
                    all_head_images.append([])

                all_head_images[head_id].extend(head_images)

        for head_id, head_images in enumerate(all_head_images):
            # The images will grow over time, so create a placeholder to be the first image (if the last image is used, the others are just pasted on top?)
            start_image = Image.new(mode="RGB", size=head_images[-1].size, color=(255, 255, 255))
            start_image.save(f'L{self._id}_head-{head_id}.gif', save_all=True, append_images=head_images, optimize=False,  # TODO: optimize?
                                        duration=10)

