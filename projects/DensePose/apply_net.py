#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import logging
import os
import sys
from typing import Any, ClassVar, Dict, List
import torch
import cv2
import numpy as np
from tqdm import tqdm

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)

DOC = """Apply Net - a tool to print / visualize DensePose results on images and videos
"""

LOGGER_NAME = "apply_net"
logger = logging.getLogger(LOGGER_NAME)

_ACTION_REGISTRY: Dict[str, "Action"] = {}


class Action:
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.",
        )


def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class InferenceAction(Action):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(InferenceAction, cls).add_arguments(parser)
        parser.add_argument("cfg", metavar="<config>", help="Config file")
        parser.add_argument("model", metavar="<model>", help="Model file")
        parser.add_argument("input", metavar="<input>", help="Input data (image, video, or directory)")
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )
        parser.add_argument(
            "--input_type",
            choices=["auto", "image", "video"],
            default="auto",
            help="Specify input type (default: auto-detect based on file extension)",
        )
        parser.add_argument(
            "--start_time",
            type=float,
            default=0.0,
            help="Start time in seconds for video processing (default: 0.0)",
        )
        parser.add_argument(
            "--end_time",
            type=float,
            default=-1.0,
            help="End time in seconds for video processing (-1 means until the end, default: -1.0)",
        )

    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        logger.info(f"Loading config from {args.cfg}")
        opts = []
        cfg = cls.setup_config(args.cfg, args.model, args, opts)
        logger.info(f"Loading model from {args.model}")
        predictor = DefaultPredictor(cfg)
        logger.info(f"Loading data from {args.input}")
        
        input_type = args.input_type
        if input_type == "auto":
            # Auto-detect based on extension
            if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                input_type = "video"
            else:
                input_type = "image"
        
        context = cls.create_context(args, cfg)
        
        if input_type == "video":
            cls._process_video(args, context, predictor)
        else:
            cls._process_images(args, context, predictor)
            
        cls.postexecute(context)

    @classmethod
    def _process_images(cls: type, args: argparse.Namespace, context: Dict[str, Any], predictor):
        file_list = cls._get_input_file_list(args.input)
        if len(file_list) == 0:
            logger.warning(f"No input images for {args.input}")
            return
            
        for file_name in file_list:
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            with torch.no_grad():
                outputs = predictor(img)["instances"]
                cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)

    @classmethod
    def _process_video(cls: type, args: argparse.Namespace, context: Dict[str, Any], predictor):
        video_path = args.input
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate start and end frames from time
        start_frame = int(args.start_time * fps)
        end_frame = int(args.end_time * fps) if args.end_time >= 0 else total_frames
        
        # Set up context for video output
        context["video_info"] = {
            "fps": fps,
            "width": width,
            "height": height,
            "output_writer": None
        }
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        frame_count = start_frame
        
        with tqdm(total=end_frame - start_frame) as pbar:
            while frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Create a file_name equivalent for video frames
                frame_file_name = f"{os.path.splitext(video_path)[0]}_frame_{frame_count:06d}"
                
                with torch.no_grad():
                    outputs = predictor(frame)["instances"]
                    cls.execute_on_outputs(
                        context, 
                        {"file_name": frame_file_name, "image": frame, "frame_idx": frame_count}, 
                        outputs
                    )
                
                frame_count += 1
                pbar.update(1)
                
        cap.release()
        
        # Close video writer if it exists
        if "video_info" in context and context["video_info"].get("output_writer") is not None:
            context["video_info"]["output_writer"].release()

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list


@register_action
class DumpAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file
    """

    COMMAND: ClassVar[str] = "dump"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Dump model outputs to a file.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="results.pkl",
            help="File name to save dump to",
        )

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        result = {"file_name": image_fpath}
        if "frame_idx" in entry:
            result["frame_idx"] = entry["frame_idx"]
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
        context["results"].append(result)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        context = {"results": [], "out_fname": args.output}
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "wb") as hFile:
            torch.save(context["results"], hFile)
            logger.info(f"Output saved to {out_fname}")


@register_action
class ShowAction(InferenceAction):
    """
    Show action that visualizes selected entries on an image or video
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Visualize selected entries")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument(
            "visualizations",
            metavar="<visualizations>",
            help="Comma separated list of visualizations, possible values: "
            "[{}]".format(",".join(sorted(cls.VISUALIZERS.keys()))),
        )
        parser.add_argument(
            "--min_score",
            metavar="<score>",
            default=0.8,
            type=float,
            help="Minimum detection score to visualize",
        )
        parser.add_argument(
            "--nms_thresh", metavar="<threshold>", default=None, type=float, help="NMS threshold"
        )
        parser.add_argument(
            "--texture_atlas",
            metavar="<texture_atlas>",
            default=None,
            help="Texture atlas file (for IUV texture transfer)",
        )
        parser.add_argument(
            "--texture_atlases_map",
            metavar="<texture_atlases_map>",
            default=None,
            help="JSON string of a dict containing texture atlas files for each mesh",
        )
        parser.add_argument(
            "--output",
            metavar="<output_file>",
            default="outputres.png",
            help="File name to save output to (will use format-specific extensions for videos)",
        )

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(args.min_score))
        if args.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(args.nms_thresh))
        cfg = super(ShowAction, cls).setup_config(config_fpath, model_fpath, args, opts)
        return cfg

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        
        # Get image and convert to grayscale if needed
        image = entry["image"]
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_vis_base = np.tile(gray_image[:, :, np.newaxis], [1, 1, 3])
        else:
            image_vis_base = image
            
        data = extractor(outputs)
        image_vis = visualizer.visualize(image_vis_base, data)
        
        # Handle video output if this is part of video processing
        if "frame_idx" in entry:
            frame_idx = entry["frame_idx"]
            # Initialize video writer if not already done
            if context["video_info"]["output_writer"] is None:
                out_fname = context["out_fname"]
                # Ensure we have a video extension
                if not out_fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    out_fname = os.path.splitext(out_fname)[0] + '.mp4'
                    context["out_fname"] = out_fname
                
                # Create output directory if it doesn't exist
                out_dir = os.path.dirname(out_fname)
                if len(out_dir) > 0 and not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                
                # Create the video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    out_fname,
                    fourcc,
                    context["video_info"]["fps"],
                    (context["video_info"]["width"], context["video_info"]["height"])
                )
                context["video_info"]["output_writer"] = video_writer
            
            # Write the frame
            context["video_info"]["output_writer"].write(image_vis)
            if frame_idx % 100 == 0:
                logger.info(f"Processed frame {frame_idx}")
        else:
            # This is an image, save it to file
            entry_idx = context["entry_idx"] + 1
            out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
            out_dir = os.path.dirname(out_fname)
            if len(out_dir) > 0 and not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(out_fname, image_vis)
            logger.info(f"Output saved to {out_fname}")
            context["entry_idx"] += 1

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        # Close video writer if it exists
        if "video_info" in context and context["video_info"].get("output_writer") is not None:
            context["video_info"]["output_writer"].release()
            logger.info(f"Video output saved to {context['out_fname']}")

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": args.output,
            "entry_idx": 0,
            "video_info": {"output_writer": None},
        }
        return context


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=120),
    )
    parser.set_defaults(func=lambda _: parser.print_help(sys.stdout))
    subparsers = parser.add_subparsers(title="Actions")
    for _, action in _ACTION_REGISTRY.items():
        action.add_parser(subparsers)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    verbosity = getattr(args, "verbosity", None)
    global logger
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(verbosity_to_level(verbosity))
    args.func(args)


if __name__ == "__main__":
    main()